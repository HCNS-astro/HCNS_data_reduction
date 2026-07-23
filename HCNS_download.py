"""
Download HST science data for the HCNS survey (proposal 18061) from MAST.

For each observed target, three product types are downloaded from the CALWF3
and CALACS pipelines:

* **DRC** – drizzle-combined mosaic, used as the dolphot reference image.
* **FLC** – CTE-corrected individual exposures, used for dolphot photometry.
* **FLT** – flat-fielded individual exposures, used when CTE correction is
  not applied (see ``CTE`` flag in ``HCNS_dolphot.py``).

Observations listed in ``bad_obs.list`` are excluded before downloading.
Individual files that have already been downloaded are skipped automatically
by ``astroquery``; new files for an existing target will be fetched on
subsequent runs.

Outputs
-------
<data_dir>/<target>/
    Downloaded FITS files, one directory per target.
HCNS_download.log
    Log of all download activity.
"""
from astroquery.mast import Observations
from astropy.table import Table
import os
import tqdm
import numpy as np


import logging
logging.basicConfig(
    filename="HCNS_download.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)


code_dir = os.getcwd()
data_dir = os.path.abspath(os.path.join(code_dir,'..','data'))


logging.info('Identifying HCNS data in MAST.')
HCNS_obs = Observations.query_criteria(proposal_id=[18061], obs_collection='HST', obs_title='Hubble Census of Nearby Satellites')

bad_obs_list = np.loadtxt('bad_obs.list',dtype='str')
# Remove flagged observations before deduplicating targets so that no data
# from bad visits is downloaded, even if the target itself is otherwise valid.
drop_inx = []
for obs_str in bad_obs_list:
    for i in range(len(HCNS_obs)):
        if obs_str.lower() in HCNS_obs['dataURL'][i]:
            drop_inx.append(i)
if len(drop_inx) > 0:
    HCNS_obs.remove_rows(drop_inx)

observed_targets = list(set(HCNS_obs['target_name']))
logging.info(f'Targets observed to date: {", ".join(observed_targets)}')


logging.info('Starting data download.')
for target in tqdm.tqdm(observed_targets):
    logging.info(f'Target: {target}')
    target_dir = os.path.join(data_dir,target)

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        logging.info(f'{target_dir} created.')

    target_obs = HCNS_obs[HCNS_obs['target_name'] == str(target)]
    target_obsids = list(set(target_obs['obsid']))

    for obsid in target_obsids:
        product_list = Observations.get_unique_product_list(str(obsid))
        # DRC: drizzle-combined mosaic (reference image for dolphot)
        filtered_products = Observations.filter_products(product_list, productType='SCIENCE', project=['CALWF3','CALACS'], productSubGroupDescription='DRC')
        logging.info(f'Downloading {", ".join(list(filtered_products["productFilename"]))}')
        Observations.download_products(filtered_products, download_dir=target_dir, flat=True)  # flat=True places files directly in target_dir
        # FLC: CTE-corrected individual exposures
        filtered_products = Observations.filter_products(product_list, productType='SCIENCE', project=['CALWF3','CALACS'], productSubGroupDescription='FLC')
        logging.info(f'Downloading {", ".join(list(filtered_products["productFilename"]))}')
        Observations.download_products(filtered_products, download_dir=target_dir, flat=True)
        # FLT: flat-fielded individual exposures (no CTE correction)
        filtered_products = Observations.filter_products(product_list, productType='SCIENCE', project=['CALWF3','CALACS'], productSubGroupDescription='FLT')
        logging.info(f'Downloading {", ".join(list(filtered_products["productFilename"]))}')
        Observations.download_products(filtered_products, download_dir=target_dir, flat=True)

