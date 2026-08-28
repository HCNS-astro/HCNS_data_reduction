"""
Download HST science data for the HCNS survey or archival programmes from MAST.

For each observed target, three product types are downloaded from the CALWF3
and CALACS pipelines:

* **DRC** – drizzle-combined mosaic, used as the dolphot reference image.
* **FLC** – CTE-corrected individual exposures, used for dolphot photometry.
* **FLT** – flat-fielded individual exposures, used when CTE correction is
  not applied (see ``CTE`` flag in ``HCNS_dolphot.py``).

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
from astropy.table import Table
import argparse
import os
import signal
import sys
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
# Shared download helper
# ---------------------------------------------------------------------------

def _download_target(target, target_dir, obs_table):
    """Download DRC, FLC and FLT products for one target."""
    os.makedirs(target_dir, exist_ok=True)
    target_obs = obs_table[obs_table['target_name'] == str(target)]
    for obsid in set(target_obs['obsid']):
        product_list = Observations.get_unique_product_list(str(obsid))
        for subgroup in ('DRC', 'FLC', 'FLT'):
            products = Observations.filter_products(
                product_list,
                productType='SCIENCE',
                project=['CALWF3', 'CALACS'],
                productSubGroupDescription=subgroup)
            logging.info(f'Downloading {", ".join(list(products["productFilename"]))}')
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

    # Load HCNS sample table for cross-matching
    google_sheet_id = '1MFvVh57tIhzc6vUUmrCvDwyYzSojJTZtpqzXfRgC48s'
    sample_url = f"https://docs.google.com/spreadsheets/d/{google_sheet_id}/export?format=csv"
    hcns_sample = pd.read_csv(sample_url, skiprows=[1])
    hcns_sample['Name'] = hcns_sample['Name'].str.upper()
    hcns_sample = hcns_sample.set_index('Name')

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
            _download_target(target, os.path.join(data_dir, target), obs)


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
        _download_target(target, os.path.join(data_dir, target), HCNS_obs)
