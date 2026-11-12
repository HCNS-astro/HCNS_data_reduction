"""
Download HST science data for the HCNS survey or archival programmes from MAST.

For each observed target, three product types are downloaded from the CALWF3
and CALACS pipelines:

* **DRC** – drizzle-combined mosaic, used as the dolphot reference image.
* **FLC** – CTE-corrected individual exposures, used for dolphot photometry.
* **FLT** – flat-fielded individual exposures, used when CTE correction is
  not applied (see ``CTE`` flag in ``HCNS_dolphot.py``).

Files already downloaded are checked against MAST: if the archive copy has
been reprocessed since (different processing time, calibration context or
software version in its primary header), the new version is downloaded in its
place. MAST reprocesses data when new calibration reference files, such as the
darks for the observation date, are delivered, often weeks after the
observations.

Flags
-----
--targets
    HCNS mode only.  Only download targets whose dataURL matches a string in
    ``good_obs.list`` (one observation ID per line).  Without this flag all
    observed targets are downloaded.
--archival
    Download archival HST data instead of the main HCNS survey.  Reads HST
    project IDs from ``archival_projects.list`` (one numeric ID per line) and
    iterates over each.  Observations are cross-matched against the HCNS
    sample table and filtered by ``archival_bad_obs.list``.  If
    ``archival_targets.list`` is non-empty, only observations whose dataURL
    matches a listed string are downloaded.

Outputs (HCNS mode)
-------------------
<data_dir>/<target>/
    Downloaded FITS files, one directory per target.
HCNS_download.log

Outputs (archival mode)
-----------------------
<data_dir>/archival/<proj_id>/<target>/
    Downloaded FITS files.
HCNS_archival_download.log
"""
from astroquery.mast import Observations
from astropy.io import fits
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import signal
import sys
import requests
import tqdm
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing (must come before logging so the filename can depend on mode)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Download HST science data for HCNS targets from MAST.')
parser.add_argument('--targets', action='store_true',
                    help='Only download targets listed in good_obs.list '
                         '(HCNS mode only).')
parser.add_argument('--archival', action='store_true',
                    help='Download archival data for projects listed in '
                         'archival_projects.list.')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

import logging
_log_file = 'HCNS_archival_download.log' if args.archival else 'HCNS_download.log'
logging.basicConfig(
    filename=_log_file,
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)


# ---------------------------------------------------------------------------
# Graceful Ctrl+C: finish the current target then stop
# ---------------------------------------------------------------------------

code_dir = os.getcwd()

_stop_after_target = False

def _request_stop(signum, frame):
    global _stop_after_target
    if not _stop_after_target:
        _stop_after_target = True
        logging.info('Shutdown requested by user (Ctrl+C).')
        tqdm.tqdm.write('\nCtrl+C received — will stop after current target finishes.')
        tqdm.tqdm.write('Press Ctrl+C again to force quit.')
    else:
        logging.warning('Forced quit by user.')
        sys.exit(1)

signal.signal(signal.SIGINT, _request_stop)


# ---------------------------------------------------------------------------
# Version check of files already downloaded
# ---------------------------------------------------------------------------

# MAST reprocesses files when STScI delivers new calibration reference files
# (e.g. the darks for an observation date), often weeks after the observations
# and usually without changing the file size, so astroquery's own check (by
# size) keeps the old version. These primary-header keywords identify the
# version instead: processing time (FLT/FLC only), processing date, calibration
# context and software versions.
VERSION_KEYS = ('PROCTIME', 'DATE', 'CRDS_CTX', 'CAL_VER', 'OPUS_VER', 'CSYS_VER')
MAST_DOWNLOAD_URL = 'https://mast.stsci.edu/api/v0.1/Download/file'
VERSION_CHECK_THREADS = 8


def _version(header):
    return {k: str(header[k]) for k in VERSION_KEYS if k in header}


def _remote_version(uri, max_bytes=2880 * 1000):
    """Version keywords of the archive copy of a product, read from the start of
    the file only (MAST serves byte ranges): the primary header ends at its END card."""
    data, pos = b'', 0
    with requests.get(MAST_DOWNLOAD_URL, params={'uri': uri}, headers={'Range': f'bytes=0-{max_bytes - 1}'},
                      stream=True, timeout=60) as response:
        response.raise_for_status()
        for chunk in response.iter_content(28800):
            data += chunk
            while pos + 80 <= len(data):
                if data[pos:pos + 3] == b'END' and not data[pos + 3:pos + 80].strip():
                    return _version(fits.Header.fromstring(data[:pos + 80].decode('ascii')))
                pos += 80
    raise ValueError(f'no END card in the first {len(data)} bytes')


def _version_change(path, uri):
    """Compare a local file with its archive copy: ``(True, changes)`` if MAST has
    a different version, ``(False, '')`` if it is the same, or ``(None, reason)``
    if the archive copy could not be checked."""
    try:
        remote = _remote_version(uri)
    except Exception as e:
        return None, f'could not read the archive copy ({e})'
    try:
        local = _version(fits.getheader(path, 0))
    except Exception as e:
        return True, f'local copy unreadable ({e})'
    changes = [f'{k} {local.get(k)} -> {remote.get(k)}' for k in VERSION_KEYS if local.get(k) != remote.get(k)]
    return bool(changes), ', '.join(changes)


def _download_again(product, target_dir):
    """Replace a local file with the archive's current version. It is downloaded
    under a temporary name first, so a failed download leaves the old file in place."""
    final = os.path.join(target_dir, product['productFilename'])
    part = final + '.part'
    status, msg, _ = Observations.download_file(product['dataURI'], local_path=part, cache=False)
    if status == 'COMPLETE' and os.path.isfile(part):
        os.replace(part, final)
        return True
    logging.error(f'{product["productFilename"]}: downloading the new version failed ({msg}); old file kept.')
    if os.path.isfile(part):
        os.remove(part)
    return False


def _update_reprocessed(target, target_dir, products):
    """Download again any product already on disk that MAST has since reprocessed."""
    existing = [p for p in products if os.path.isfile(os.path.join(target_dir, p['productFilename']))]
    if not existing:
        return
    with ThreadPoolExecutor(VERSION_CHECK_THREADS) as pool:
        results = list(pool.map(lambda p: _version_change(os.path.join(target_dir, p['productFilename']),
                                                          p['dataURI']), existing))
    n_new = 0
    for product, (changed, detail) in zip(existing, results):
        name = product['productFilename']
        if changed is None:
            logging.warning(f'{target}: {name}: {detail}; kept, unless its size differs.')
        elif changed:
            logging.info(f'{target}: {name}: MAST has a new version ({detail}); downloading it again.')
            n_new += _download_again(product, target_dir)
    if n_new:
        tqdm.tqdm.write(f'{target}: {n_new} file(s) reprocessed by MAST since they were downloaded; '
                        f'new versions downloaded.')


# ---------------------------------------------------------------------------
# Shared download helper
# ---------------------------------------------------------------------------

def _download_target(target, target_dir, target_obs, exclude=()):
    """Download DRC, FLC and FLT products for one target.

    ``target_obs`` must already be filtered to the rows for this target.
    ``exclude`` is an iterable of observation-ID substrings (e.g. from
    bad_obs.list); any product whose filename contains one of these strings
    is skipped.  This is needed in addition to any upstream row-level
    filtering because MAST can expose the same exposure files as products of
    a different (e.g. HAP-association) observation record whose dataURL
    doesn't literally name the exposure.
    """
    for obsid in set(target_obs['obsid']):
        product_list = Observations.get_unique_product_list(str(obsid))
        for subgroup in ('DRC', 'FLC', 'FLT'):
            products = Observations.filter_products(
                product_list,
                productType='SCIENCE',
                project=['CALWF3', 'CALACS'],
                productSubGroupDescription=subgroup)
            if len(exclude) > 0:
                keep_inx = [i for i in range(len(products))
                            if not any(s.lower() in products['productFilename'][i].lower()
                                       for s in exclude)]
                if len(keep_inx) < len(products):
                    dropped = set(products['productFilename']) - set(products[keep_inx]['productFilename'])
                    logging.info(f'Excluding bad-obs products: {", ".join(sorted(dropped))}')
                products = products[keep_inx]
            if len(products) == 0:
                continue
            logging.info(f'Downloading {", ".join(list(products["productFilename"]))}')
            # Created only once there is something to download, so a target
            # whose products were all excluded doesn't leave an empty dir.
            os.makedirs(target_dir, exist_ok=True)
            # Files already here are replaced if MAST has reprocessed them; astroquery
            # then downloads any that are missing (or whose size differs) and skips the rest.
            _update_reprocessed(target, target_dir, products)
            Observations.download_products(products, download_dir=target_dir, flat=True)


# ---------------------------------------------------------------------------
# ARCHIVAL mode
# ---------------------------------------------------------------------------

if args.archival:
    # Load project list
    proj_ids = np.loadtxt('archival_projects.list', dtype='str', ndmin=1).tolist()
    if not proj_ids:
        logging.warning('archival_projects.list is empty. Nothing to do.')
        sys.exit(0)

    # Load bad-obs exclusion list
    archival_bad = np.loadtxt('archival_bad_obs.list', dtype='str', ndmin=1)

    # Load optional target filter (non-empty → restrict to listed obs IDs)
    archival_targets = np.loadtxt('archival_targets.list', dtype='str', ndmin=1)
    if len(archival_targets) > 0:
        logging.info(f'archival_targets.list: will restrict to {len(archival_targets)} listed observation(s).')

    # Load optional MAST→HCNS name map (two columns: MAST_NAME HCNS_NAME)
    name_map = {}
    if os.path.isfile('archival_name_map.list'):
        with open('archival_name_map.list') as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#'):
                    _mast, _hcns = _line.split()
                    name_map[_mast.upper()] = _hcns.upper()
    if name_map:
        logging.info(f'archival_name_map.list: {len(name_map)} name mapping(s) loaded.')

    # Load HCNS sample table for cross-matching
    google_sheet_id = '1lsoONcIf1X_LWHUwfktuWNVAg2whnhH9tc9s0fjhQ84'
    sample_url = f"https://docs.google.com/spreadsheets/d/{google_sheet_id}/export?format=csv"
    hcns_sample = pd.read_csv(sample_url, skiprows=[1])
    hcns_sample['Name'] = hcns_sample['Name'].str.upper()
    hcns_sample = hcns_sample.set_index('Name')

    # -----------------------------------------------------------------
    # TSV-driven downloads: targeted per-row observation codes for
    # archival dwarfs identified in HCNS_Archival_Dwarfs.tsv.  Each row
    # already specifies exact observation codes, so no HCNS-sample
    # cross-match is needed here (unlike the whole-project scan below).
    # Runs first since these are quick, targeted downloads.
    # -----------------------------------------------------------------
    tsv_path = 'HCNS_Archival_Dwarfs.tsv'
    if os.path.isfile(tsv_path):
        tsv = pd.read_csv(tsv_path, sep='\t')
        usable = tsv[tsv['Codes (Observation ID)'].notna() &
                    (tsv['Codes (Observation ID)'].str.strip() != '')]
        logging.info(f'{tsv_path}: {len(usable)} usable row(s) found.')

        # Load existing name-map entries so we don't duplicate them.
        existing_map_entries = set()
        if os.path.isfile('archival_name_map.list'):
            with open('archival_name_map.list') as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith('#'):
                        existing_map_entries.add(_line.split()[0].upper())
        new_map_lines = []

        proj_obs_cache = {}
        for _, row in usable.iterrows():
            if _stop_after_target:
                logging.info('Download stopped by user. Remaining tsv rows skipped.')
                break

            hcns_name = str(row['HCNS_Name']).strip()
            other_name = str(row['Other_Names']).strip() if pd.notna(row['Other_Names']) else hcns_name
            codes = [c.strip().lower() for c in str(row['Codes (Observation ID)']).split(',') if c.strip()]
            proj_id = str(int(row['Proposal_ID']))

            if proj_id not in proj_obs_cache:
                logging.info(f'Querying MAST for project {proj_id} (tsv-driven).')
                proj_obs = Observations.query_criteria(proposal_id=[int(proj_id)], obs_collection='HST')
                drop_inx = [i for i in range(len(proj_obs))
                            if any(s.lower() in proj_obs['dataURL'][i] for s in archival_bad)]
                if drop_inx:
                    proj_obs.remove_rows(drop_inx)
                proj_obs_cache[proj_id] = proj_obs
            proj_obs = proj_obs_cache[proj_id]

            keep_inx = [i for i in range(len(proj_obs))
                        if any(code in proj_obs['dataURL'][i] for code in codes)]
            if not keep_inx:
                logging.warning(f'{hcns_name}: none of the codes {codes} matched any '
                                f'observation in project {proj_id}. Skipping.')
                continue
            row_obs = proj_obs[keep_inx]

            mast_names = sorted(set(row_obs['target_name']))
            if other_name.upper() not in [n.upper() for n in mast_names]:
                logging.info(f'{hcns_name}: MAST target_name {mast_names} differs from '
                            f'tsv Other_Names "{other_name}" (see {tsv_path}).')

            logging.info(f'{hcns_name} ({other_name}): downloading proposal {proj_id}, codes {codes}.')
            data_dir_tsv = os.path.abspath(os.path.join(code_dir, '..', 'data', 'archival', proj_id))
            os.makedirs(data_dir_tsv, exist_ok=True)
            _download_target(hcns_name, os.path.join(data_dir_tsv, hcns_name), row_obs,
                             exclude=archival_bad)

            if other_name.upper() != hcns_name.upper() and other_name.upper() not in existing_map_entries:
                new_map_lines.append(f'{other_name.upper()}\t{hcns_name.upper()}\n')
                existing_map_entries.add(other_name.upper())

        if new_map_lines:
            with open('archival_name_map.list', 'a') as f:
                f.writelines(new_map_lines)
            logging.info(f'Appended {len(new_map_lines)} new entries to archival_name_map.list.')

    # -----------------------------------------------------------------
    # Whole-project scan: iterate archival_projects.list, cross-matching
    # each project's full observation list against the HCNS sample.
    # -----------------------------------------------------------------
    for proj_id in proj_ids:
        if _stop_after_target:
            logging.info('Download stopped by user. Remaining projects skipped.')
            break

        logging.info(f'Querying MAST for project {proj_id}.')
        obs = Observations.query_criteria(
            proposal_id=[int(proj_id)], obs_collection='HST')

        # Apply bad-obs exclusions
        drop_inx = [i for i in range(len(obs))
                    if any(s.lower() in obs['dataURL'][i] for s in archival_bad)]
        if drop_inx:
            obs.remove_rows(drop_inx)

        # Normalise MAST target names: apply name map where available,
        # otherwise just uppercase.  This makes the cross-match case-insensitive
        # and lets the name map handle MAST naming-convention differences.
        obs['target_name'] = [name_map.get(t.upper(), t.upper())
                              for t in obs['target_name']]

        # Keep only targets in the HCNS sample
        keep_inx = [i for i in range(len(obs))
                    if obs[i]['target_name'] in hcns_sample.index]
        rejected = set(obs['target_name']) - set(obs[keep_inx]['target_name'])
        if rejected:
            logging.info(f'{proj_id}: rejected (not in HCNS sample): {", ".join(sorted(rejected))}')
        obs = obs[keep_inx]

        # Apply archival_targets.list filter if non-empty
        if len(archival_targets) > 0:
            keep_inx = [i for i in range(len(obs))
                        if any(s.lower() in obs['dataURL'][i]
                               for s in archival_targets)]
            obs = obs[keep_inx]
            logging.info(f'{proj_id}: {len(keep_inx)} observations match archival_targets.list.')

        observed_targets = list(set(obs['target_name']))
        logging.info(f'{proj_id}: targets to download: {", ".join(observed_targets)}')

        data_dir = os.path.abspath(
            os.path.join(code_dir, '..', 'data', 'archival', proj_id))
        os.makedirs(data_dir, exist_ok=True)

        logging.info(f'Starting download for project {proj_id}.')
        for target in tqdm.tqdm(observed_targets, desc=proj_id):
            if _stop_after_target:
                logging.info('Download stopped by user. Remaining targets skipped.')
                break
            logging.info(f'Target: {target}')
            _download_target(target, os.path.join(data_dir, target),
                             obs[obs['target_name'] == str(target)],
                             exclude=archival_bad)


# ---------------------------------------------------------------------------
# HCNS mode (default)
# ---------------------------------------------------------------------------

else:
    data_dir = os.path.abspath(os.path.join(code_dir, '..', 'data'))

    logging.info('Identifying HCNS data in MAST.')
    HCNS_obs = Observations.query_criteria(
        proposal_id=[18061], obs_collection='HST',
        obs_title='Hubble Census of Nearby Satellites')

    bad_obs_list = np.loadtxt('bad_obs.list', dtype='str')
    # Remove flagged observations before deduplicating targets so that no data
    # from bad visits is downloaded, even if the target itself is otherwise valid.
    drop_inx = []
    for obs_str in bad_obs_list:
        for i in range(len(HCNS_obs)):
            if obs_str.lower() in HCNS_obs['dataURL'][i]:
                drop_inx.append(i)
    if len(drop_inx) > 0:
        HCNS_obs.remove_rows(drop_inx)

    if args.targets:
        good_obs_list = np.loadtxt('good_obs.list', dtype='str', ndmin=1)
        keep_inx = [i for i in range(len(HCNS_obs))
                    if any(obs_str.lower() in HCNS_obs['dataURL'][i]
                           for obs_str in good_obs_list)]
        HCNS_obs = HCNS_obs[keep_inx]
        logging.info(f'--targets active: {len(keep_inx)} observations match good_obs.list.')

    observed_targets = list(set(HCNS_obs['target_name']))
    logging.info(f'Targets to download: {", ".join(observed_targets)}')

    logging.info('Starting data download.')
    for target in tqdm.tqdm(observed_targets):
        if _stop_after_target:
            logging.info('Download stopped by user. Remaining targets skipped.')
            break
        logging.info(f'Target: {target}')
        _download_target(target, os.path.join(data_dir, target),
                         HCNS_obs[HCNS_obs['target_name'] == str(target)],
                         exclude=bad_obs_list)
