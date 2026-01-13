from astroquery.mast import Observations
from astropy.table import Table
import os
import tqdm


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

observed_targets = list(set(HCNS_obs['target_name']))
logging.info(f'Targets observed to date: {", ".join(observed_targets)}')


logging.info('Starting data download.')
for target in tqdm.tqdm(observed_targets):
    logging.info(f'Target: {target}')
    target_dir = os.path.join(data_dir,target)

    if os.path.isdir(target_dir):
        logging.info(f'{target_dir} already exists.')
        logging.warning(f'{target} will be skipped.')
        logging.info('If you wish to re-download these data, you must first rename or delete this directory.')
    else:
        os.mkdir(target_dir)
        logging.info(f'{target_dir} created.')

        target_obs = HCNS_obs[HCNS_obs['target_name'] == str(target)]
        target_obsids = list(set(target_obs['obsid']))

        for obsid in target_obsids:
            product_list = Observations.get_unique_product_list(str(obsid))
            filtered_products = Observations.filter_products(product_list, productType='SCIENCE', project='CALWF3', productSubGroupDescription='DRC')
            logging.info(f'Downloading {", ".join(list(filtered_products['productFilename']))}')
            Observations.download_products(filtered_products, download_dir=target_dir, flat=True)
            filtered_products = Observations.filter_products(product_list, productType='SCIENCE', project='CALWF3', productSubGroupDescription='FLT')
            logging.info(f'Downloading {", ".join(list(filtered_products['productFilename']))}')
            Observations.download_products(filtered_products, download_dir=target_dir, flat=True)

