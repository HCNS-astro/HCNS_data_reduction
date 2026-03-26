import sys, os, glob, shutil, logging
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval, LinearStretch, make_rgb, ManualInterval
from astropy.wcs import WCS
from reproject import reproject_adaptive
import astroalign

verbose = False

def make_logger(name, filename, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(filename, encoding="utf-8", mode="a")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = make_logger("RGB_images", filename="HCNS_RGB_images.log")


code_dir = os.getcwd()
data_dir = os.path.abspath(os.path.join(code_dir,'..','data'))
out_dir = os.path.abspath(os.path.join(code_dir,'..','output'))
reduct_dir = os.path.abspath(os.path.join(code_dir,'..','reduction'))


paths = glob.glob(os.path.join(data_dir,"*"))
for path in paths:
    target = str(os.path.split(path)[1])
    logger.info(f'Making images for {target}.')
    
    target_dir = os.path.join(data_dir,target)
    dolphot_dir = os.path.join(reduct_dir,target)
    target_out_dir = os.path.join(out_dir,target)
    
    if os.path.isdir(target_out_dir):
        if verbose:
            logger.info(f'{target_out_dir} already exists.')
    else:
        os.mkdir(target_out_dir)
        logger.info(f'{target_out_dir} created.')
    
    drizfilelist = glob.glob(os.path.join(target_dir,'*drc.fits'))
    
    filters = []
    
    for imgpath in drizfilelist:
        hdu = fits.open(imgpath)
        #header = hdu[0].header
        header = hdu[0].header
        filtername = header['FILTER']
        filters.append(filtername)
        hdu.close()
    filters = list(set(filters))
    filters.sort()
    
    logger.info(f'Available filters for {target}: {", ".join(filters)}')
    
    filterdrizimg = []
    
    for i,filtername in enumerate(filters):
        for imgpath in drizfilelist:
            imgfile = os.path.split(imgpath)[1]
            inx = imgfile.find('.fits')
            rootname = imgfile[:inx]
            hdu = fits.open(imgpath)
            #header = hdu[0].header
            header = hdu[0].header
            if filtername in header['FILTER']:
                filterdrizimg.append(rootname)
            hdu.close()
    
    
    if len(filters) <= 1:
        logger.error('An RGB image cannot be made with fewer than 2 filters.')
        logger.info(f'Skipping {target}.')
    if len(filters) == 2:
        logger.info(f'Filters for RGB image will be: {filters[1]}, {filters[0]}+{filters[1]}, {filters[0]}')
    else:
        logger.error('More than two filters is not currently supported.')
        logger.info(f'Skipping {target}.')
    
    
    imgpath = os.path.join(target_dir,filterdrizimg[0]+'.fits')
    hdu = fits.open(imgpath)
    blue_header = hdu[1].header
    blue_img = hdu[1].data
    blue_img = blue_img.view(blue_img.dtype.newbyteorder()).byteswap()
    blue_wcs = WCS(blue_header)
    hdu.close()
    
    imgpath = os.path.join(target_dir,filterdrizimg[1]+'.fits')
    hdu = fits.open(imgpath)
    red_header = hdu[1].header
    red_img = hdu[1].data
    red_img = red_img.view(red_img.dtype.newbyteorder()).byteswap()
    red_wcs = WCS(red_header)
    hdu.close()
    
    
    interval = ZScaleInterval(contrast=0.35,max_iterations=5)
    interval.get_limits(red_img)
    norm = ImageNormalize(red_img, interval=interval, stretch=LinearStretch())
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(projection=red_wcs)
    plt.imshow(red_img, norm=norm, origin='lower', cmap='Greys', aspect='equal')
    plt.axis('off')
    plt.savefig(os.path.join(target_out_dir,f'{target}_{filters[1]}_greyscale.png'),bbox_inches='tight')
    logger.info(f'Red image saved to: {target}_{filters[1]}_greyscale.png')
    plt.close()
    
    
    interval = ZScaleInterval(contrast=0.35,max_iterations=5)
    interval.get_limits(blue_img)
    norm = ImageNormalize(blue_img, interval=interval, stretch=LinearStretch())
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(projection=red_wcs)
    plt.imshow(blue_img, norm=norm, origin='lower', cmap='Greys', aspect='equal')
    plt.axis('off')
    plt.savefig(os.path.join(target_out_dir,f'{target}_{filters[0]}_greyscale.png'),bbox_inches='tight')
    logger.info(f'Blue image saved to: {target}_{filters[0]}_greyscale.png')
    plt.close()
    
    
    try:
        blue_img, footprint = astroalign.register(blue_img, red_img, detection_sigma=4, min_area=9)
        blue_img, footprint = reproject_adaptive((blue_img,red_wcs), red_wcs, shape_out=np.shape(red_img))
    except astroalign.MaxIterError:
        logger.warning(f"WARNING: Astroalign failed for {target}. Falling back to header WCS.")
        blue_img, footprint = reproject_adaptive((blue_img,blue_wcs), red_wcs, shape_out=np.shape(red_img))
    
    
    RGB_img = make_rgb(red_img,0.5*(red_img+blue_img),blue_img, interval=ManualInterval(vmin=0, vmax=0.03))
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(projection=red_wcs)
    plt.imshow(RGB_img, origin='lower', aspect='equal')
    plt.axis('off')
    plt.savefig(os.path.join(target_out_dir,f'{target}_RGB.png'),bbox_inches='tight')
    logger.info(f'RGB image saved to: {target}_RGB.png')
    plt.close()