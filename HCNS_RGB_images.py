import sys, os, glob, shutil, logging, copy
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval, LinearStretch, make_rgb, ManualInterval
from astropy.wcs import WCS
from reproject import reproject_adaptive
import astroalign
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u
from scipy.spatial import KDTree
from skimage.transform import AffineTransform
import cv2

verbose = False

def icp_affine(src_pts, dst_pts, max_iterations=50, tolerance=0.001):
    """
    Iterative Closest Point to find an affine transform without known correspondence.
    (This function was mostly authored by Gemini 3.0 - 2026-03-26)
    """
    src = np.copy(src_pts)
    dst_tree = KDTree(dst_pts)
    
    prev_error = 0
    
    for i in range(max_iterations):
        # 1. Find the nearest neighbor in dst_pts for each point in src
        distances, indices = dst_tree.query(src)
        matched_dst = dst_pts[indices]
        
        # 2. Estimate Affine Transform (using RANSAC to handle your 'not in both' points)
        matrix, inliers = cv2.estimateAffine2D(src, matched_dst, method=cv2.RANSAC)
        
        if matrix is None: break
        
        # 3. Apply the transform to src points for the next iteration
        # Add a row of ones for matrix multiplication: [x, y, 1]
        src_homo = np.hstack([src, np.ones((src.shape[0], 1))])
        src = (matrix @ src_homo.T).T
        
        # Check convergence
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Convert to 3x3 matrix format of astroalign
    full_matrix = np.vstack([matrix, [0, 0, 1]])
    at_object = AffineTransform(matrix=full_matrix)

    return at_object

def star_cloud_alignment(red_img, red_header, red_wcs, blue_img, blue_header, blue_wcs, max_points=50):

    #Select stars from red image
    mean, median, std = sigma_clipped_stats(red_img, sigma=3.0)
    threshold = 5.0 * std
    daofind = DAOStarFinder(threshold, fwhm=3, sharplo=0.2, sharphi=1.5)
    sources = daofind(red_img - median)
    sources['mag'] = sources['mag'] - red_header['PHOTZPT']
    sources.sort('mag')
    red_srcs = sources[sources['mag'] > 15.]
    red_srcs = red_srcs[red_srcs['mag'] < 25.]
    red_coords = pixel_to_skycoord(red_srcs['xcentroid'],red_srcs['ycentroid'],red_wcs)
    red_srcs['ra'] = red_coords.ra.deg
    red_srcs['dec'] = red_coords.dec.deg

    #Select stars from the blue image
    mean, median, std = sigma_clipped_stats(blue_img, sigma=3.0)
    threshold = 5.0 * std
    daofind = DAOStarFinder(threshold, fwhm=3, sharplo=0.2, sharphi=1.5)
    sources = daofind(blue_img - median)
    sources['mag'] = sources['mag'] - blue_header['PHOTZPT']
    sources.sort('mag')
    blue_srcs = sources[sources['mag'] > 15.]
    blue_srcs = blue_srcs[blue_srcs['mag'] < 25.]
    blue_coords = pixel_to_skycoord(blue_srcs['xcentroid'],blue_srcs['ycentroid'],blue_wcs)
    blue_srcs['ra'] = blue_coords.ra.deg
    blue_srcs['dec'] = blue_coords.dec.deg

    #Approximately match stars using WCS
    idx, d2d, d3d = red_coords.match_to_catalog_sky(blue_coords)
    match_srcs = copy.deepcopy(red_srcs)
    match_srcs['ra_blue'] = blue_srcs[idx]['ra']
    match_srcs['dec_blue'] = blue_srcs[idx]['dec']
    match_srcs['xcentroid_blue'] = blue_srcs[idx]['xcentroid']
    match_srcs['ycentroid_blue'] = blue_srcs[idx]['ycentroid']
    match_srcs['mag_blue'] = blue_srcs[idx]['mag']
    match_srcs['separation'] = d2d.arcsec
    match_srcs['color'] = match_srcs['mag_blue'] - match_srcs['mag']

    #Remove clearly bad matches
    match_srcs = match_srcs[match_srcs['separation'] < 0.5] #arcsec
    match_srcs = match_srcs[match_srcs['color'] < 1.75] #Too red
    match_srcs = match_srcs[match_srcs['color'] > -0.25] #Too blue

    #Perform point cloud alignment
    affine_trans = icp_affine(np.array(list(zip(match_srcs[0:max_points]['xcentroid'],match_srcs[0:max_points]['ycentroid']))),
                              np.array(list(zip(match_srcs[0:max_points]['xcentroid_blue'],match_srcs[0:max_points]['ycentroid_blue']))))

    return affine_trans

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
    
    target_dir = os.path.join(data_dir,target)
    dolphot_dir = os.path.join(reduct_dir,target)
    target_out_dir = os.path.join(out_dir,target)
    
    if os.path.isfile(os.path.join(target_out_dir,f'{target}_RGB.png')):
        if verbose:
            logger.info(f'Images already exist and will not be regenerated.')
        continue
    else:
        logger.info(f'Making images for {target}.')
        if not os.path.isdir(target_out_dir):
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
    plt.savefig(os.path.join(target_out_dir,f'{target}_{filters[1]}_greyscale.png'),bbox_inches='tight',dpi=200)
    logger.info(f'Red image saved to: {target}_{filters[1]}_greyscale.png')
    plt.close()
    
    
    interval = ZScaleInterval(contrast=0.35,max_iterations=5)
    interval.get_limits(blue_img)
    norm = ImageNormalize(blue_img, interval=interval, stretch=LinearStretch())
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(projection=red_wcs)
    plt.imshow(blue_img, norm=norm, origin='lower', cmap='Greys', aspect='equal')
    plt.axis('off')
    plt.savefig(os.path.join(target_out_dir,f'{target}_{filters[0]}_greyscale.png'),bbox_inches='tight',dpi=200)
    logger.info(f'Blue image saved to: {target}_{filters[0]}_greyscale.png')
    plt.close()
    
    
    try:
        blue_img, footprint = astroalign.register(blue_img, red_img, detection_sigma=4, min_area=9)
        blue_img, footprint = reproject_adaptive((blue_img,red_wcs), red_wcs, shape_out=np.shape(red_img))
    except astroalign.MaxIterError:
        logger.warning(f"WARNING: Astroalign failed for {target}. Attempting star cloud alignment.")
        try:
            aa_transform = star_cloud_alignment(red_img, red_header, red_wcs, blue_img, blue_header, blue_wcs)
            registered_blue_img, footprint = astroalign.apply_transform(aa_transform, blue_img, red_img)
            blue_img, footprint = reproject_adaptive((blue_img,red_wcs), red_wcs, shape_out=np.shape(red_img))
        except:
            logger.warning(f"WARNING: Star cloud alignment failed for {target}. Falling back to header WCS.")
            blue_img, footprint = reproject_adaptive((blue_img,blue_wcs), red_wcs, shape_out=np.shape(red_img))
    
    
    RGB_img = make_rgb(red_img,0.5*(red_img+blue_img),blue_img, interval=ManualInterval(vmin=0, vmax=0.03))
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(projection=red_wcs)
    plt.imshow(RGB_img, origin='lower', aspect='equal')
    plt.axis('off')
    plt.savefig(os.path.join(target_out_dir,f'{target}_RGB.png'),bbox_inches='tight',dpi=200)
    logger.info(f'RGB image saved to: {target}_RGB.png')
    plt.close()