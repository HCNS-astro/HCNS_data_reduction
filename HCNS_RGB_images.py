"""
Create greyscale and RGB composite images for HCNS survey targets.

For each target, the script reads the drizzled DRC images for both available
filters, removes hot pixels and cosmic rays (outliers with no counterpart in
any other filter; ``--no-clean`` turns this off), aligns the blue-filter image
to the red-filter frame (trying astroalign first, then a star-cloud ICP method,
then WCS-only reprojection), and saves a three-colour composite PNG alongside
individual greyscale PNGs. The composite has the same colour stretch for every
target, on a fixed surface-brightness scale, so that the images of different
targets are directly comparable.

Targets are processed in parallel, ``--ncpu`` at a time. Each target's log
messages are written as one block when it finishes, and a target that fails is
logged and skipped without stopping the others.

Outputs (per target, under ``out_dir/<target>/``)
-------------------------------------------------
<target>_<blue_filter>_greyscale.png
<target>_<red_filter>_greyscale.png
<target>_RGB.png
HCNS_RGB_images.log
"""
import argparse
import sys, os, glob, shutil, logging, copy
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.visualization import (ImageNormalize, ZScaleInterval, LinearStretch, AsinhStretch,
                                   make_rgb, ManualInterval)
from astropy.wcs import WCS
from reproject import reproject_adaptive
import astroalign
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.wcs.utils import pixel_to_skycoord, proj_plane_pixel_area
import astropy.units as u
from scipy.spatial import KDTree
from scipy.ndimage import (median_filter, maximum_filter, gaussian_filter,
                           binary_propagation, binary_dilation)
from skimage.transform import AffineTransform, SimilarityTransform
import cv2
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Create RGB images for HCNS targets.')
parser.add_argument('--overwrite', action='store_true',
                    help='Regenerate images even if they already exist.')
parser.add_argument('--ncpu', type=int, default=10,
                    help='Number of targets processed simultaneously (default: 10). Each one holds its '
                         'images in memory, so lower this if memory runs short.')
parser.add_argument('--no-clean', action='store_true',
                    help='Use the DRC images as they are, without removing hot pixels and cosmic rays.')
args = parser.parse_args()

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
    total = np.eye(3)  # product of every iteration's correction
    
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
        total = np.vstack([matrix, [0, 0, 1]]) @ total
        
        # Check convergence
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    if np.array_equal(total, np.eye(3)):
        raise RuntimeError('ICP found no transform.')
    at_object = AffineTransform(matrix=total)

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
    affine_trans = icp_affine(np.array(list(zip(match_srcs[0:max_points]['xcentroid_blue'],match_srcs[0:max_points]['ycentroid_blue']))),
                              np.array(list(zip(match_srcs[0:max_points]['xcentroid'],match_srcs[0:max_points]['ycentroid']))))

    return affine_trans


# astroalign detection settings, tried in order until one finds a transform. The
# strict settings go first so that hot pixels and cosmic rays don't crowd the
# stars out of the control points (the looser ones fail on DW0317M4142).
ALIGN_ATTEMPTS = [dict(detection_sigma=10, min_area=25),
                  dict(detection_sigma=4,  min_area=9)]


def find_transform_to_reference(sec_img, sec_header, sec_wcs, ref_img, ref_header, ref_wcs,
                                target, filtername, logger):
    """Find the transform from a secondary filter's pixel grid to the reference's.

    Two tiers: 1. astroalign (triangle-matching), trying each of
    ``ALIGN_ATTEMPTS``; 2. star-cloud ICP if astroalign fails.

    Returns
    -------
    skimage.transform transform or None
        Maps secondary pixel coordinates to reference pixel coordinates;
        ``None`` if both methods fail (``align_to_reference`` then falls back
        to the header WCS).
    """
    for kwargs in ALIGN_ATTEMPTS:
        try:
            transform, _ = astroalign.find_transform(sec_img, ref_img, **kwargs)
            return transform
        except Exception as e:
            error = e
    logger.warning(f"WARNING: Astroalign failed for {target} ({filtername}): {error}. Attempting star cloud alignment.")
    try:
        return star_cloud_alignment(ref_img, ref_header, ref_wcs, sec_img, sec_header, sec_wcs)
    except Exception:
        logger.warning(f"WARNING: Star cloud alignment failed for {target} ({filtername}). Falling back to header WCS.")
        return None


def align_to_reference(sec_img, sec_wcs, ref_img, ref_wcs, transform):
    """Align a secondary-filter image onto the reference image's pixel grid,
    with ``transform`` from ``find_transform_to_reference``, or with the
    header WCS alone if that is ``None``."""
    if transform is None:
        aligned, footprint = reproject_adaptive((sec_img, sec_wcs), ref_wcs, shape_out=np.shape(ref_img))
    else:
        aligned, footprint = astroalign.apply_transform(transform, sec_img, ref_img)
    return aligned


# ---------------------------------------------------------------------------
# Hot-pixel and cosmic-ray removal (turned off with --no-clean). With only two
# exposures per filter, the drizzle leaves many warm pixels and cosmic rays in
# the DRC images. They land in one exposure of one filter, while real sources
# appear in every filter, so an outlier with no counterpart in any other
# filter is removed. Tuned with test_hotpixel.py on DW0317M4142 (WFC3) and
# DW0132P1422 (ACS).
# ---------------------------------------------------------------------------
CLEAN_DETECT_SIGMA = 4.0   # outliers above this (residual from the local median) are candidates
CLEAN_GROW_SIGMA   = 2.0   # candidates grow into connected pixels above this
CLEAN_MATCH_SIGMA  = 2.5   # another filter above this within CLEAN_MATCH_BOX -> real source
CLEAN_MATCH_BOX    = 3     # tolerance (px) for sub-pixel misalignment between filters
CLEAN_MATCH_FRAC   = 0.15  # ...and that filter's peak within CLEAN_PEAK_BOX must reach this
CLEAN_PEAK_BOX     = 5     #    fraction of this filter's peak (in sigma), else no real source
CLEAN_MEDIAN_BOX   = 5     # local-median box (px) for the residual map
CLEAN_FILL_SIGMA   = 1.5   # Gaussian kernel (px) used to fill removed pixels


def _significance(img, good, std=None):
    """Residual from the local median, in units of ``std``: by default the
    clipped std of the residual over ``good`` pixels.  Returns (significance, std)."""
    residual = img - median_filter(img, size=CLEAN_MEDIAN_BOX)
    if std is None:
        _, _, std = sigma_clipped_stats(residual[good], sigma=3)
    return residual / std, std


def _no_counterpart(sig_self, sig_other):
    """Pixels with no counterpart in the other filter: either nothing significant
    within CLEAN_MATCH_BOX, or nothing bright enough nearby to match this
    filter's peak (no real source is that extreme in colour; hot pixels and
    cosmic rays often land next to a faint source in the other filter)."""
    faint   = maximum_filter(sig_other, size=CLEAN_MATCH_BOX) < CLEAN_MATCH_SIGMA
    extreme = (maximum_filter(sig_other, size=CLEAN_PEAK_BOX)
               < CLEAN_MATCH_FRAC * maximum_filter(sig_self, size=CLEAN_PEAK_BOX))
    return faint | extreme


def _fill_masked(img, mask, good):
    """Replace masked pixels with the Gaussian-weighted mean of the unmasked
    good pixels around them (normalized convolution)."""
    weight = (good & ~mask).astype(float)
    num    = gaussian_filter(img * weight, CLEAN_FILL_SIGMA)
    den    = gaussian_filter(weight, CLEAN_FILL_SIGMA)
    result = img.copy()
    result[mask] = num[mask] / np.maximum(den[mask], 1e-6)
    return result


def _isolated_rejected(img, weight):
    """On-chip pixels with zero weight (every exposure rejected), except the
    strips of them along the chip edges and gap, which are coverage edges
    rather than artifacts: filling those in one filter but not another would
    leave single-colour fringes."""
    finite   = np.isfinite(img)
    rejected = finite & (weight == 0)
    edges    = binary_propagation(binary_dilation(~finite) & rejected, mask=rejected)
    return rejected & ~edges


def remove_artifacts(images, weights, transforms, logger):
    """Remove hot pixels and cosmic rays from each filter's image, in its own
    pixel grid.

    Every other filter is warped onto this filter's grid. Pixels more than
    ``CLEAN_DETECT_SIGMA`` above the local median that have no counterpart in
    any other filter are flagged, grown into connected pixels above
    ``CLEAN_GROW_SIGMA`` (cosmic-ray tracks, drizzle spill-over) and by one
    more pixel, then filled from the unflagged pixels around them. Pixels
    where another filter has no data are never flagged. Isolated on-chip
    pixels with zero weight (every exposure rejected) are filled too.
    Non-finite (off-chip) pixels are left as they are.

    Parameters
    ----------
    images : dict
        ``{filtername: (img, header, wcs)}`` as loaded from the DRC files.
    weights : dict
        ``{filtername: WHT array}``.
    transforms : dict
        ``{filtername: transform}`` from each filter's pixel grid to the
        reference grid (identity for the reference filter); these must be
        accurate to a pixel or so, or real sources lose their counterparts.
    logger : logging.Logger

    Returns
    -------
    dict
        ``{filtername: cleaned img}``, same shape and dtype as the input.
    """
    data, good, sig = {}, {}, {}
    for f, (img, _, _) in images.items():
        finite  = np.isfinite(img)
        data[f] = np.where(finite, img, 0.0).astype(float)
        good[f] = finite & (weights[f] > 0)
        sig[f], _ = _significance(data[f], good[f])

    eight = np.ones((3, 3), bool)
    cleaned = {}
    for f, (img, _, _) in images.items():
        lonely = good[f].copy()
        for g in images:
            if g == f:
                continue
            g_to_f = transforms[g] + transforms[f].inverse
            warped, _ = astroalign.apply_transform(g_to_f, data[g], data[f], fill_value=0)
            cover, _  = astroalign.apply_transform(g_to_f, good[g].astype(float), data[f], fill_value=0)
            covered   = cover > 0.5
            sig_g, _  = _significance(warped, good[f] & covered)
            lonely &= covered & _no_counterpart(sig[f], sig_g)

        seeds = (sig[f] > CLEAN_DETECT_SIGMA) & lonely
        grow  = (sig[f] > CLEAN_GROW_SIGMA) & lonely
        mask  = binary_propagation(seeds, structure=eight, mask=grow)
        mask  = binary_dilation(mask, structure=eight) & lonely
        mask |= _isolated_rejected(img, weights[f])

        filled = _fill_masked(data[f], mask, good[f])
        cleaned[f] = img.copy()
        cleaned[f][mask] = filled[mask]
        logger.info(f'{f}: {seeds.sum()} outlier pixels with no counterpart in the other filter(s); '
                    f'{mask.sum()} pixels ({100 * mask.mean():.1f}%) filled, including their surroundings.')
    return cleaned


# ---------------------------------------------------------------------------
# RGB colour stretch, the same for every target so that the images are directly
# comparable (if one galaxy looks brighter than another, it is). Each filter is
# converted to surface brightness, f_lambda per arcsec^2, in units of
# RGB_SB_UNIT: the typical sky noise of the HCNS images (F606W and F814W, ACS
# and WFC3 are all within 25% of it, so the sky looks alike in every image and
# its noise is close to grey). An asinh stretch then maps RGB_BLACK..RGB_WHITE
# to 0..1: the sky (0) goes to 0.06, 5 units to 0.45, and RGB_WHITE (about
# 20 AB mag/arcsec^2 at 7000 A) to white. Finally the colour saturation is
# boosted by RGB_SATURATION. Chosen by comparing stretches on the HCNS targets
# and a 3-filter ACS field.
# ---------------------------------------------------------------------------
RGB_SB_UNIT    = 4.4e-19  # erg/s/cm^2/A/arcsec^2
RGB_BLACK      = -0.449   # in RGB_SB_UNIT
RGB_WHITE      = 50       # in RGB_SB_UNIT
RGB_SOFTENING  = 1.855    # in RGB_SB_UNIT: the stretch is linear below this, logarithmic above
RGB_SATURATION = 1.6


def _surface_brightness(img, header):
    """Sky-subtracted surface brightness of an image, in units of RGB_SB_UNIT.

    ``header`` is the image's own SCI header (PHOTFLAM and the native pixel
    size; an aligned image keeps its native values per pixel). Non-finite
    (off-chip) pixels are set to the sky level."""
    finite = np.isfinite(img) & (img != 0)
    _, sky, _ = sigma_clipped_stats(img[finite][::7], sigma=3, maxiters=10)
    pixel_area = proj_plane_pixel_area(WCS(header)) * 3600 ** 2   # deg^2 -> arcsec^2
    sb = (img - sky) * header['PHOTFLAM'] / pixel_area / RGB_SB_UNIT
    return np.where(np.isfinite(sb), sb, 0.0)


def _boost_saturation(rgb, factor):
    """Scale each pixel's departure from grey (the mean of its channels) by ``factor``."""
    grey = rgb.mean(axis=-1, keepdims=True)
    return np.clip(grey + factor * (rgb - grey), 0, 1)


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

class _RecordCollector(logging.Handler):
    """Logging handler that keeps records in a list, so that a worker can
    send one target's log messages back to the main process."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        # Bake any traceback into the message text and drop the exc_info,
        # which cannot be pickled, so the record can be sent between processes.
        record.msg = self.format(record)
        record.args = None
        record.exc_info = None
        record.exc_text = None
        self.records.append(record)

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

def _process_target(eff_data_dir, eff_out_dir, target):
    """Make the greyscale and RGB images for one target.

    Parameters
    ----------
    eff_data_dir, eff_out_dir : str
        Data and output directories holding this target's subdirectory
        (the HCNS ones, or one archival program's).
    target : str
        Target name.
    """
    target_dir = os.path.join(eff_data_dir,target)
    target_out_dir = os.path.join(eff_out_dir,target)
    
    if os.path.isfile(os.path.join(target_out_dir,f'{target}_RGB.png')) and not args.overwrite:
        rgb_mtime = os.path.getmtime(os.path.join(target_out_dir, f'{target}_RGB.png'))
        data_files = glob.glob(os.path.join(target_dir, '*.fits'))
        if data_files and max(os.path.getmtime(f) for f in data_files) > rgb_mtime:
            logger.warning(
                f'{target}: input data files are newer than {target}_RGB.png. '
                f'Delete the output files to trigger regeneration.'
            )
        if verbose:
            logger.info(f'Images already exist and will not be regenerated.')
        return
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
                    logger.error('No filter identified.')
        filters.append(filtername)
        hdu.close()
    # Longest wavelength (reddest) first; this ordering sets the RGB reference
    # image (index 0) and channel assignment below.
    filters = sorted(set(filters), key=lambda f: int(f[1:4]), reverse=True)

    logger.info(f'Available filters for {target}: {", ".join(filters)}')

    filterdrizimg = {}

    for filtername in filters:
        for imgpath in drizfilelist:
            imgfile = os.path.split(imgpath)[1]
            inx = imgfile.find('.fits')
            rootname = imgfile[:inx]
            hdu = fits.open(imgpath)
            header = hdu[0].header
            match instrument:
                case 'WFC3':
                    if filtername in header['FILTER']:
                        filterdrizimg[filtername] = rootname
                case 'ACS':
                    if filtername in header['FILTER1']:
                        filterdrizimg[filtername] = rootname
                    elif filtername in header['FILTER2']:
                        filterdrizimg[filtername] = rootname
            hdu.close()


    if len(filters) < 2 or len(filters) > 3:
        logger.warning(f'{target} has data in {len(filters)} filters. RGB image creation supports 2 or 3. Skipping.')
        return
    logger.info(f'Filters for RGB image (red to blue): {", ".join(filters)}')


    images = {}
    weights = {}
    for filtername in filters:
        imgpath = os.path.join(target_dir, filterdrizimg[filtername]+'.fits')
        hdu = fits.open(imgpath)
        header = hdu[1].header
        img = hdu[1].data
        img = img.view(img.dtype.newbyteorder()).byteswap()  # native byte order for astroalign/cv2
        wcs = WCS(header)
        weights[filtername] = hdu['WHT'].data
        hdu.close()
        images[filtername] = (img, header, wcs)


    # Transform from every filter's pixel grid to the reddest (reference)
    # filter's.
    ref_filter = filters[0]
    ref_img, ref_header, ref_wcs = images[ref_filter]
    transforms = {ref_filter: SimilarityTransform()}
    for filtername in filters[1:]:
        sec_img, sec_header, sec_wcs = images[filtername]
        transforms[filtername] = find_transform_to_reference(
            sec_img, sec_header, sec_wcs, ref_img, ref_header, ref_wcs, target, filtername, logger)


    # Remove hot pixels and cosmic rays, each filter in its own pixel grid. This
    # needs every filter's transform to within a pixel or so, which the header
    # WCS alone is not always (6 px off between the filters of DW0317M4142).
    if args.no_clean:
        logger.info('Hot pixels and cosmic rays not removed (--no-clean).')
    elif any(transform is None for transform in transforms.values()):
        logger.warning(f'WARNING: Hot pixels and cosmic rays not removed for {target}: some filter '
                       f'could only be aligned by its header WCS, which is not accurate enough to '
                       f'match sources between filters.')
    else:
        logger.info('Removing hot pixels and cosmic rays.')
        cleaned = remove_artifacts(images, weights, transforms, logger)
        images = {f: (cleaned[f], h, w) for f, (_, h, w) in images.items()}
        ref_img = images[ref_filter][0]


    # Save one greyscale PNG per filter, each in its own native pixel/WCS frame.
    for filtername in filters:
        img, header, wcs = images[filtername]
        interval = ZScaleInterval(contrast=0.35,max_iterations=5)
        interval.get_limits(img)
        norm = ImageNormalize(img, interval=interval, stretch=LinearStretch())

        fig = plt.figure(figsize=(15,15))
        ax = plt.subplot(projection=wcs)
        plt.imshow(img, norm=norm, origin='lower', cmap='Greys', aspect='equal')
        plt.axis('off')
        plt.savefig(os.path.join(target_out_dir,f'{target}_{filtername}_greyscale.png'),bbox_inches='tight',dpi=200)
        logger.info(f'{filtername} image saved to: {target}_{filtername}_greyscale.png')
        plt.close()


    # Align every non-reference filter onto the reddest (reference) filter's
    # pixel grid.
    aligned = {ref_filter: ref_img}
    for filtername in filters[1:]:
        sec_img, sec_header, sec_wcs = images[filtername]
        aligned[filtername] = align_to_reference(sec_img, sec_wcs, ref_img, ref_wcs,
                                                 transforms[filtername])


    # Red to blue, on the fixed surface-brightness scale (see RGB_SB_UNIT).
    channels = [_surface_brightness(aligned[f], images[f][1]) for f in filters]
    if len(filters) == 2:
        # Green channel synthesised as the average of red and blue since only
        # two science filters are available.
        channels.insert(1, 0.5 * (channels[0] + channels[1]))
    RGB_img = make_rgb(*channels, interval=ManualInterval(vmin=RGB_BLACK, vmax=RGB_WHITE),
                       stretch=AsinhStretch(a=RGB_SOFTENING / (RGB_WHITE - RGB_BLACK)),
                       output_dtype=np.float64)
    RGB_img = _boost_saturation(RGB_img, RGB_SATURATION)

    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(projection=ref_wcs)
    plt.imshow(RGB_img, origin='lower', aspect='equal')
    plt.axis('off')
    plt.savefig(os.path.join(target_out_dir,f'{target}_RGB.png'),bbox_inches='tight',dpi=200)
    logger.info(f'RGB image saved to: {target}_RGB.png')
    plt.close()


def _run_target(eff_data_dir, eff_out_dir, target):
    """Run ``_process_target`` for one target, collecting its log messages.

    This runs in a joblib worker process, which does not execute this
    script's top-level code: the logger there has no handlers and matplotlib
    has its default backend, so both are set up here. Any error is logged and
    the target skipped, so that it does not stop the other targets.

    Returns
    -------
    str, str
        ``target`` and ``eff_out_dir``.
    list of logging.LogRecord
        The target's log messages, for the main process to write out as one
        block.
    bool
        ``False`` if processing raised an exception.
    """
    target_logger = logging.getLogger("RGB_images")
    saved = target_logger.handlers[:], target_logger.level, target_logger.propagate
    collector = _RecordCollector()
    target_logger.handlers = [collector]
    target_logger.setLevel(logging.INFO)
    target_logger.propagate = False
    ok = True
    try:
        matplotlib.use('Agg')
        _process_target(eff_data_dir, eff_out_dir, target)
    except Exception:
        target_logger.exception(f'Making images for {target} ({eff_out_dir}) failed; skipping it.')
        ok = False
    finally:
        plt.close('all')
        target_logger.handlers, level, target_logger.propagate = saved
        target_logger.setLevel(level)
    return target, eff_out_dir, collector.records, ok


logger.info(f'Processing {len(all_targets)} target(s), up to {args.ncpu} at a time.')
failed = []
results = Parallel(n_jobs=args.ncpu, return_as='generator_unordered')(
    delayed(_run_target)(*entry) for entry in all_targets)
for target, eff_out_dir, records, ok in results:
    # Write each target's messages together, as soon as that target is done.
    for record in records:
        logger.handle(record)
    if not ok:
        failed.append(f'{target} ({eff_out_dir})')
if failed:
    logger.warning(f'{len(failed)} target(s) failed and were skipped (tracebacks above): '
                   f'{", ".join(failed)}')