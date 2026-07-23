"""
Create greyscale and RGB composite images for HCNS survey targets.

For each target, the script reads the drizzled DRC images for both available
filters, aligns the blue-filter image to the red-filter frame (trying
astroalign first, then a star-cloud ICP method, then WCS-only reprojection),
and saves a three-colour composite PNG alongside individual greyscale PNGs.

Outputs (per target, under ``out_dir/<target>/``)
-------------------------------------------------
<target>_<blue_filter>_greyscale.png
<target>_<red_filter>_greyscale.png
<target>_RGB.png
HCNS_RGB_images.log
"""
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
    """Estimate an affine transform between two point clouds via ICP.

    Iterative Closest Point algorithm that does not require pre-established
    point correspondences.  At each iteration, nearest neighbours are found
    and RANSAC is used to robustly fit the affine transform, so points present
    in one cloud but not the other do not corrupt the estimate.

    Originally authored with assistance from Gemini 3.0 (2026-03-26).

    Parameters
    ----------
    src_pts : numpy.ndarray, shape (N, 2)
        Source point coordinates (x, y) to be transformed.
    dst_pts : numpy.ndarray, shape (M, 2)
        Destination point coordinates (x, y).
    max_iterations : int, optional
        Maximum number of ICP iterations.  Default is ``50``.
    tolerance : float, optional
        Convergence threshold: stop when the change in mean nearest-neighbour
        distance falls below this value.  Default is ``0.001``.

    Returns
    -------
    skimage.transform.AffineTransform
        Affine transform mapping ``src_pts`` coordinates to ``dst_pts``.
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
    """Align a blue-filter image to a red-filter image using detected stars.

    Detects point sources in both images with DAOStarFinder, matches them by
    sky coordinate using the image WCS, removes outliers by separation and
    colour, then calls ``icp_affine`` to compute the pixel-space affine
    transform.  Used as a fallback when ``astroalign`` fails.

    Parameters
    ----------
    red_img : numpy.ndarray
        2-D science image array for the red filter.
    red_header : astropy.io.fits.Header
        FITS header for the red image; must contain ``'PHOTZPT'``.
    red_wcs : astropy.wcs.WCS
        World Coordinate System for the red image.
    blue_img : numpy.ndarray
        2-D science image array for the blue filter.
    blue_header : astropy.io.fits.Header
        FITS header for the blue image; must contain ``'PHOTZPT'``.
    blue_wcs : astropy.wcs.WCS
        World Coordinate System for the blue image.
    max_points : int, optional
        Maximum number of matched stars passed to ``icp_affine``.
        Default is ``50``.

    Returns
    -------
    skimage.transform.AffineTransform
        Affine transform that maps the blue image pixel frame to the red
        image pixel frame.
    """
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
    """Create a logger that writes to both a file and stdout.

    Parameters
    ----------
    name : str
        Name identifier for the logger instance.
    filename : str
        Path to the log file (opened in append mode).
    level : int, optional
        Logging level threshold; default is ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger with file and console handlers attached.
    """
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


# HCNS targets -- exclude the 'archival' subdirectory
all_targets = [(data_dir, out_dir, os.path.basename(p))
               for p in glob.glob(os.path.join(data_dir, '*'))
               if os.path.isdir(p) and os.path.basename(p) != 'archival']

# Archival targets -- scan data/archival/<prog>/<target> for DRC files
archival_data_base = os.path.abspath(os.path.join(code_dir, '..', 'data',   'archival'))
archival_out_base  = os.path.abspath(os.path.join(code_dir, '..', 'output', 'archival'))
for prog_dir in sorted(glob.glob(os.path.join(archival_data_base, '*'))):
    prog_id = os.path.basename(prog_dir)
    for target_path in sorted(glob.glob(os.path.join(prog_dir, '*'))):
        if os.path.isdir(target_path):
            all_targets.append((
                prog_dir,
                os.path.join(archival_out_base, prog_id),
                os.path.basename(target_path),
            ))

for eff_data_dir, eff_out_dir, target in all_targets:
    target_dir = os.path.join(eff_data_dir,target)
    target_out_dir = os.path.join(eff_out_dir,target)
    
    if os.path.isfile(os.path.join(target_out_dir,f'{target}_RGB.png')):
        if verbose:
            logger.info(f'Images already exist and will not be regenerated.')
        continue
    else:
        logger.info(f'Making images for {target}.')
        os.makedirs(target_out_dir, exist_ok=True)
    
    drizfilelist = glob.glob(os.path.join(target_dir,'*drc.fits'))
    
    filters = []
    instrument = None
    for imgpath in drizfilelist:
        hdu = fits.open(imgpath)
        header = hdu[0].header
        if 'ACS' in header['INSTRUME']:
            instrument = 'ACS'
        elif 'WFC3' in header['INSTRUME']:
            instrument = 'WFC3'
        else:
            logger.warning(f'Instrument not set for {target}. Skipping RGB image creation.')
            continue
        match instrument:
            case 'WFC3':
                filtername = header['FILTER']
            case 'ACS':
                if 'CLEAR' not in header['FILTER1']:
                    filtername = header['FILTER1']
                elif 'CLEAR' not in header['FILTER2']:
                    filtername = header['FILTER2']
                else:
                    global_logger.error('No filter identified.')
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
            header = hdu[0].header
            match instrument:
                case 'WFC3':
                    if filtername in header['FILTER']:
                        filterdrizimg.append(rootname)
                case 'ACS':
                    if filtername in header['FILTER1']:
                        filterdrizimg.append(rootname)
                    elif filtername in header['FILTER2']:
                        filterdrizimg.append(rootname)
            hdu.close()
    
    
    if len(filters) < 2:
        logger.warning(f'{target} has data in fewer than 2 filters ({len(filters)} found). Skipping RGB image creation.')
        continue
    if len(filters) > 2:
        logger.warning(f'{target} has data in more than 2 filters ({len(filters)} found). RGB image creation not supported. Skipping.')
        continue
    logger.info(f'Filters for RGB image will be: {filters[1]}, {filters[0]}+{filters[1]}, {filters[0]}')
    
    
    imgpath = os.path.join(target_dir,filterdrizimg[0]+'.fits')
    hdu = fits.open(imgpath)
    blue_header = hdu[1].header
    blue_img = hdu[1].data
    blue_img = blue_img.view(blue_img.dtype.newbyteorder()).byteswap()  # native byte order for astroalign/cv2
    blue_wcs = WCS(blue_header)
    hdu.close()

    imgpath = os.path.join(target_dir,filterdrizimg[1]+'.fits')
    hdu = fits.open(imgpath)
    red_header = hdu[1].header
    red_img = hdu[1].data
    red_img = red_img.view(red_img.dtype.newbyteorder()).byteswap()  # native byte order for astroalign/cv2
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
    
    
    # Align the blue image to the red pixel grid using a three-tier fallback:
    # 1. astroalign (triangle-matching); 2. star-cloud ICP if astroalign raises
    # MaxIterError; 3. WCS-only reprojection if both alignment methods fail.
    try:
        blue_img, footprint = astroalign.register(blue_img, red_img, detection_sigma=4, min_area=9)
        blue_img, footprint = reproject_adaptive((blue_img,red_wcs), red_wcs, shape_out=np.shape(red_img))
    except astroalign.MaxIterError:
        logger.warning(f"WARNING: Astroalign failed for {target}. Attempting star cloud alignment.")
        try:
            aa_transform = star_cloud_alignment(red_img, red_header, red_wcs, blue_img, blue_header, blue_wcs)
            registered_blue_img, footprint = astroalign.apply_transform(aa_transform, blue_img, red_img)
            blue_img, footprint = reproject_adaptive((registered_blue_img,red_wcs), red_wcs, shape_out=np.shape(red_img))
        except:
            logger.warning(f"WARNING: Star cloud alignment failed for {target}. Falling back to header WCS.")
            blue_img, footprint = reproject_adaptive((blue_img,blue_wcs), red_wcs, shape_out=np.shape(red_img))
    
    
    # Green channel synthesised as the average of red and blue since only two science filters are available.
    RGB_img = make_rgb(red_img,0.5*(red_img+blue_img),blue_img, interval=ManualInterval(vmin=0, vmax=0.03))
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(projection=red_wcs)
    plt.imshow(RGB_img, origin='lower', aspect='equal')
    plt.axis('off')
    plt.savefig(os.path.join(target_out_dir,f'{target}_RGB.png'),bbox_inches='tight',dpi=200)
    logger.info(f'RGB image saved to: {target}_RGB.png')
    plt.close()