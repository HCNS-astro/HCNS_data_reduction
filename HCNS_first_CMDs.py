"""
Process dolphot photometry output and AST results to produce initial CMDs.

For each target with completed dolphot photometry, this script:

* Reads the dolphot output catalog and applies quality cuts (source type,
  photometry flags, magnitude, crowding, sharpness).
* Queries the SFD dust map to compute per-star extinction corrections.
* Saves a full-field and a target-region (within 2 r_e) photometry catalog.
* Produces colour–magnitude diagrams as PDF figures.

If AST results are also available, the script fits completeness curves as a
function of F814W magnitude and F606W–F814W colour and saves the parameters
to ``completeness.dat``.

Outputs (per target, under ``out_dir/<target>/``)
-------------------------------------------------
phot_full.csv
    Full-field extinction-corrected photometry catalog.
phot_target_initial.csv
    Photometry within 2 r_e of the target centre.
CMD_full.pdf, CMD_initial.pdf
    Colour–magnitude diagrams.
phot_ast.csv
    Recovered fake-star catalog from ASTs.
completeness.pdf, completeness.dat
    Completeness limit curves and best-fit model parameters.
HCNS_first_CMDs.log
    Run log.
"""
import argparse
import os, sys, glob, numpy, scipy, pandas
import shutil, subprocess, logging
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from dustmaps.sfd import SFDQuery


plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 18
})


parser = argparse.ArgumentParser(description='Generate initial CMDs for HCNS targets.')
parser.add_argument('--overwrite', action='store_true',
                    help='Reprocess outputs even if they already exist.')
args = parser.parse_args()

code_dir = os.getcwd()
data_dir = os.path.abspath(os.path.join(code_dir,'..','data'))
out_dir = os.path.abspath(os.path.join(code_dir,'..','output'))
reduct_dir = os.path.abspath(os.path.join(code_dir,'..','reduction'))


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

def close_logger(logger_instance):
    """Flush and remove all handlers from a logger instance.

    Parameters
    ----------
    logger_instance : logging.Logger
        The logger to shut down.
    """
    handlers = logger_instance.handlers[:]
    for handler in handlers:
        handler.close()
        logger_instance.removeHandler(handler)


global_logger = make_logger("global", filename="HCNS_first_CMDs.log")

def comp_func(x, x50, wid):
    """Complementary error function model for photometric completeness.

    Returns the fraction of stars recovered at magnitude ``x``, modelled as a
    smoothed step function that falls from 1 at bright magnitudes to 0 at
    faint magnitudes.

    Parameters
    ----------
    x : float or array-like
        Magnitude(s) at which to evaluate completeness.
    x50 : float
        50% completeness magnitude (mid-point of the transition).
    wid : float
        Width parameter controlling the steepness of the transition.

    Returns
    -------
    float or numpy.ndarray
        Completeness fraction in the range [0, 1].
    """
    return 0.5*(1.-scipy.special.erf((x-x50)/(wid*numpy.sqrt(2.))))

def inv_comp_func(c, x50, wid):
    """Inverse of ``comp_func``: convert a completeness fraction to a magnitude.

    Parameters
    ----------
    c : float or array-like
        Completeness fraction(s) in the range (0, 1).
    x50 : float
        50% completeness magnitude.
    wid : float
        Width parameter of the completeness model.

    Returns
    -------
    float or numpy.ndarray
        Magnitude(s) corresponding to completeness fraction ``c``.
    """
    return x50 + wid*numpy.sqrt(2.)*scipy.special.erfinv(1.-2.*c)

def col_comp_func(col, tran, plat, alpha):
    """Piecewise completeness-limit model as a function of stellar colour.

    Below the transition colour ``tran`` the limit is constant at ``plat``.
    Above ``tran`` it rises quadratically, modelling the increasing difficulty
    of detecting red stars against a redder sky background.

    Parameters
    ----------
    col : float or array-like
        Colour value(s) (e.g. F606W − F814W).
    tran : float
        Transition colour below which the limit is flat.
    plat : float
        Constant completeness-limit magnitude for ``col < tran``.
    alpha : float
        Quadratic coefficient governing the rise above ``tran``.

    Returns
    -------
    float or numpy.ndarray
        Completeness-limit magnitude as a function of colour.
    """
    return numpy.where(col < tran, plat, alpha*col**2 - 2.*alpha*tran*col + plat + alpha*tran**2.)
    

R_WFC3_F814W = 1.536 # WFC3 values from Schlafly and Finkbeiner (2011)
R_WFC3_F606W = 2.488
R_WFC3_F555W = 2.855
R_WFC3_F475W = 3.248
R_ACS_F814W =  1.526 # ACS values from Schlafly and Finkbeiner (2011)
R_ACS_F606W = 2.471
R_ACS_F555W = 2.792
R_ACS_F475W = 3.268
R_I = 1.505 # Landolt values from Schlafly and Finkbeiner (2011)
R_V = 2.742
R_B = 3.626

# Looked up by (instrument, filter) so the correct coefficient is applied
# regardless of which blue filter HCNS_dolphot.py's BLUE_FILTER_PREFERENCE
# actually selected for a given target (F606W, F555W, or F475W).
EXTINCTION_COEFFS = {
    ('ACS', 'F814W'): R_ACS_F814W, ('ACS', 'F606W'): R_ACS_F606W,
    ('ACS', 'F555W'): R_ACS_F555W, ('ACS', 'F475W'): R_ACS_F475W,
    ('WFC3', 'F814W'): R_WFC3_F814W, ('WFC3', 'F606W'): R_WFC3_F606W,
    ('WFC3', 'F555W'): R_WFC3_F555W, ('WFC3', 'F475W'): R_WFC3_F475W,
}

# Dolphot's "Transformed UBVRI magnitude" column doesn't say which Johnson
# letter it corresponds to -- that's a dolphot convention, not written
# anywhere in the .columns file. Looked up here instead.
JOHNSON_LETTER = {
    'F606W': 'V',
    'F555W': 'V',
    'F475W': 'B',
    'F814W': 'I',
}
JOHNSON_EXTINCTION = {'V': R_V, 'B': R_B, 'I': R_I}

max_mag = 30.
max_sharp = 0.1
crowd_thresh = 1.0


def _parse_dolphot_columns(columns_file):
    """Parse a dolphot ``.columns`` file into a rename map plus filter info.

    Parameters
    ----------
    columns_file : str
        Path to the ``.columns`` file dolphot writes alongside its
        photometry output.

    Returns
    -------
    dict
        Maps 0-indexed normal-format column position to a short name:
        ``'x'``, ``'y'``, ``'SNR_global'``, ``'type'``, and per filter
        ``'{filt}_mag'``, ``'{filt}_{letter}vega'`` (or ``'{filt}_UBVRI'`` if
        the filter has no entry in ``JOHNSON_LETTER``), ``'e_{filt}'``,
        ``'SNR_{filt}'``, ``'sharp_{filt}'``, ``'crowd_{filt}'``.
        Unrecognised columns (chi, extension, chip, ...) are absent from
        the map.
    list of str
        Combined-block filters found, in the order dolphot reported them.
    list of str
        One entry per per-exposure image slot (length = number of images
        dolphot used, i.e. ``Nimg``), giving that slot's filter, in the same
        img1...imgN order dolphot declared them.
    """
    if not os.path.isfile(columns_file):
        raise FileNotFoundError(
            f'{columns_file} not found; cannot auto-detect dolphot columns.')

    GLOBAL_PREFIXES = {
        'Object X position': 'x',
        'Object Y position': 'y',
        'Object type': 'type',
    }
    FILTER_FIELDS = {
        'Instrumental VEGAMAG magnitude': 'mag',
        'Transformed UBVRI magnitude': 'vega',
        'Magnitude uncertainty': 'e',
        'Signal-to-noise': 'SNR',
        'Sharpness': 'sharp',
        'Crowding': 'crowd',
    }

    rename = {}
    filters_found = []
    image_filters = []
    with open(columns_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_str, sep, desc = line.partition('.')
            if not sep or not num_str.strip().isdigit():
                continue
            col_index = int(num_str.strip()) - 1
            desc = desc.strip()

            # Global signal-to-noise has no comma suffix; distinguish from
            # the per-filter "Signal-to-noise, <inst>_<filter>" below.
            if desc == 'Signal-to-noise':
                rename[col_index] = 'SNR_global'
                continue

            matched_global = False
            for prefix, short in GLOBAL_PREFIXES.items():
                if desc.startswith(prefix):
                    rename[col_index] = short
                    matched_global = True
                    break
            if matched_global:
                continue

            # Per-exposure columns carry a "(<filter>, <exptime> sec)"
            # suffix; only the combined per-filter blocks (no parentheses)
            # are used for the rename map, but each per-exposure block's
            # first column marks a new image slot -- record its filter.
            if '(' in desc:
                if desc.startswith('Measured counts'):
                    filt_tag = desc.rpartition('(')[2].split(',')[0].strip()
                    _, _, filtername = filt_tag.rpartition('_')
                    image_filters.append(filtername)
                continue
            if ', ' not in desc:
                continue

            field_desc, _, filt_tag = desc.rpartition(', ')
            if '_' not in filt_tag or field_desc not in FILTER_FIELDS:
                continue
            _, _, filtername = filt_tag.rpartition('_')

            if filtername not in filters_found:
                filters_found.append(filtername)

            short = FILTER_FIELDS[field_desc]
            if short == 'mag':
                name = f'{filtername}_mag'
            elif short == 'vega':
                letter = JOHNSON_LETTER.get(filtername)
                if letter is None:
                    global_logger.warning(
                        f'No Johnson-letter mapping for {filtername}; '
                        f'using generic "{filtername}_UBVRI" column name.')
                    name = f'{filtername}_UBVRI'
                else:
                    name = f'{filtername}_{letter}vega'
            else:
                name = f'{short}_{filtername}'
            rename[col_index] = name

    if not filters_found:
        raise ValueError(f'No filter columns found in {columns_file}.')

    return rename, filters_found, image_filters


def _read_dolphot_catalog(catalog_path):
    """Read a dolphot photometry catalog, auto-detecting column names from
    the companion ``<catalog_path>.columns`` file dolphot writes alongside
    its output.

    Column positions -- and which filters are present -- can change between
    dolphot versions or if the pipeline is ever run with more than two
    filters, so nothing about the layout is hardcoded here.

    Parameters
    ----------
    catalog_path : str
        Path to the dolphot photometry output file (no extension).

    Returns
    -------
    pandas.DataFrame
        Catalog with columns renamed per ``_parse_dolphot_columns``.
    list of str
        Filters found, in the order dolphot reported them.
    """
    rename, filters_found, _ = _parse_dolphot_columns(catalog_path + '.columns')
    df = pandas.read_csv(catalog_path, sep=r'\s+', header=None)
    df = df.rename(columns=rename)
    return df, filters_found


def _read_fake_catalog(fake_path, catalog_path):
    """Read a dolphot artificial-star (``.fake``) catalog, auto-detecting
    column names from the companion ``<catalog_path>.columns`` file (the
    *normal* photometry catalog's columns file -- per the dolphot manual,
    ".fake" output uses the same format as normal photometry, except that
    the true position (on the reference frame) and true brightness (on
    each image) are prepended to the start of the line").

    Parameters
    ----------
    fake_path : str
        Path to the ``.fake`` file to read.
    catalog_path : str
        Path to the corresponding normal photometry output file (no
        extension) -- used only to locate ``<catalog_path>.columns``.

    Returns
    -------
    pandas.DataFrame
        Columns ``'x'``, ``'y'`` (true position), ``'{filt}_in'`` (true
        injected magnitude) and ``'{filt}_mag'``/``'SNR_{filt}'`` (recovered
        photometry) per filter, ``'SNR'`` (global), and ``'recovered'`` (1 if
        the star passed the same quality cuts as the main catalog, else 0).
    list of str
        Filters found, in the order dolphot reported them.
    """
    rename, filters_found, image_filters = _parse_dolphot_columns(catalog_path + '.columns')
    n_images = len(image_filters)
    offset = 4 + 2 * n_images

    df_raw = pandas.read_csv(fake_path, sep=r'\s+', header=None)

    df = pandas.DataFrame(index=df_raw.index)
    for col_index, name in rename.items():
        df[name] = df_raw[col_index + offset]
    for filt in filters_found:
        first_slot = image_filters.index(filt)
        df[f'{filt}_in'] = df_raw[4 + 2*first_slot + 1]
    # True position (columns 2, 3) is set last: 'x'/'y' are also produced by
    # the recovered-column rename above (dolphot's own X/Y fields), which
    # must not win over the true position here.
    df['x'] = df_raw[2] - 0.5
    df['y'] = df_raw[3] - 0.5

    condition = numpy.ones(len(df), dtype=bool)
    for filt in filters_found:
        condition &= (df[f'{filt}_mag'] < 99.)
    condition &= (df['type'] < 3)
    condition &= (sum(df[f'crowd_{filt}'] for filt in filters_found) < crowd_thresh)
    condition &= (sum(df[f'sharp_{filt}'] for filt in filters_found)**2. < max_sharp)
    df['recovered'] = numpy.where(condition, 1, 0)
    df['SNR'] = df['SNR_global']

    return df, filters_found


def _fit_completeness(fake_stars, blue_filter, red_filter, out_dir, target, suffix='', min_snr=None, log_suffix=''):
    """Fit color-dependent completeness curves from an AST catalog and save
    ``completeness{suffix}.dat``/``completeness{suffix}.pdf``.

    Parameters
    ----------
    fake_stars : pandas.DataFrame
        AST catalog, as returned by ``_read_fake_catalog`` (or a
        concatenation of several such catalogs).
    blue_filter, red_filter : str
        Filter names used for the colour and reference magnitude.
    out_dir : str
        Root output directory (target's own subdirectory is appended).
    target : str
        Target identifier, used for titles, filenames, and log messages.
    suffix : str, optional
        Appended to the output filenames, e.g. ``'_4sig'``. Default ``''``.
    min_snr : float or None, optional
        If given, a star only counts as recovered when it also satisfies
        ``SNR_{blue_filter} >= min_snr`` and ``SNR_{red_filter} >= min_snr``,
        in addition to the existing ``recovered`` criteria. The denominator
        (total injected stars per bin) is unchanged -- only the recovered
        count becomes stricter. Default ``None`` (no additional cut).
    log_suffix : str, optional
        Appended to log messages (e.g. ``' (full AST catalog)'``).
        Default ``''``.
    """
    m_min = 21.
    m_max = 30.
    m_wid = 0.1
    m_bins = numpy.arange(m_min, m_max+0.1*m_wid, m_wid)

    colmax = 2.0
    colmin = -1.0
    colwid = 0.2
    colbins = numpy.arange(colmin, colmax+0.1*colwid, colwid)

    C90 = numpy.zeros(len(colbins)-1)
    C50 = numpy.zeros(len(colbins)-1)

    if min_snr is not None:
        recovered = ((fake_stars['recovered'] > 0) &
                    (fake_stars[f'SNR_{blue_filter}'] >= min_snr) &
                    (fake_stars[f'SNR_{red_filter}'] >= min_snr))
    else:
        recovered = fake_stars['recovered'] > 0
    mag_in = fake_stars[f'{red_filter}_in']
    color_in = fake_stars[f'{blue_filter}_in'] - fake_stars[f'{red_filter}_in']

    global_logger.info(f'Fitting completeness limits for {target}{log_suffix}.')
    for c in range(len(colbins)-1):
        condition = (color_in < colbins[c+1]) & (color_in > colbins[c])
        fake_stars_colbin = fake_stars[condition]

        comp = numpy.zeros(len(m_bins)-1)
        cnts = numpy.zeros(len(m_bins)-1)

        for i in fake_stars_colbin.index:
            j = int(max(min(len(m_bins)-2, numpy.floor((mag_in[i]-m_min)/m_wid)), 0))
            if recovered[i]:
                comp[j] += 1.
            cnts[j] += 1.

        comp = comp/cnts
        inx = numpy.where(comp > 0.4)[0]

        try:
            erf_fit = scipy.optimize.curve_fit(comp_func, m_bins[inx]+0.05, comp[inx], p0=[26.5,0.5],
                                               sigma=1./numpy.sqrt(cnts[inx]), bounds=[[24.,0.1],[28.,3.]])
            C90[c] = inv_comp_func(0.9, erf_fit[0][0], erf_fit[0][1])
            C50[c] = erf_fit[0][0]
        except ValueError:
            C90[c] = numpy.nan
            C50[c] = numpy.nan

    try:
        fit90 = scipy.optimize.curve_fit(col_comp_func, colbins[:-1]+0.5*colwid, C90, p0=[0.8,26.5,0.])
        global_logger.info(f"90% Completeness parameters: [{fit90[0][0]}, {fit90[0][1]}, {fit90[0][2]}]")

        # 50% completeness varies nearly linearly with colour over this range, so a
        # simple linear model is used rather than the piecewise model applied to 90%.
        fit50 = scipy.optimize.curve_fit(lambda x,a,b: a*x + b, colbins[:-1]+0.5*colwid, C50, p0=[1,30])
        global_logger.info(f"50% Completeness parameters: [{fit50[0][0]}, {fit50[0][1]}]")

        with open(os.path.join(out_dir, target, f'completeness{suffix}.dat'), 'w') as f:
            f.write(f'comp50 = [{fit50[0][0]}, {fit50[0][1]}]\n')
            f.write(f'comp90 = [{fit90[0][0]}, {fit90[0][1]}, {fit90[0][2]}]')

        x_tmp = numpy.arange(colmin, colmax, 0.01)
        plt.plot(x_tmp, col_comp_func(x_tmp, fit90[0][0], fit90[0][1], fit90[0][2]))
        plt.plot(x_tmp, fit50[0][0]*x_tmp + fit50[0][1])
    except (ValueError, RuntimeError):
        global_logger.warning(f'Completeness limit fit failed for {target}{log_suffix}.')

    plt.scatter(colbins[:-1]+0.5*colwid, C90)
    plt.scatter(colbins[:-1]+0.5*colwid, C50)
    plt.ylim(29, 24)
    plt.ylabel(red_filter)
    plt.xlabel(f'{blue_filter}-{red_filter}')
    plt.savefig(os.path.join(out_dir, target, f'completeness{suffix}.pdf'), bbox_inches='tight')
    plt.close()


# HCNS targets -- exclude the 'archival' subdirectory
all_targets = [(data_dir, reduct_dir, out_dir, os.path.basename(p))
               for p in glob.glob(os.path.join(data_dir, '*'))
               if os.path.isdir(p) and os.path.basename(p) != 'archival']

# Archival targets -- scan reduction/archival/<prog>/<target> for reduced data
archival_data_base   = os.path.abspath(os.path.join(code_dir, '..', 'data',      'archival'))
archival_reduct_base = os.path.abspath(os.path.join(code_dir, '..', 'reduction', 'archival'))
archival_out_base    = os.path.abspath(os.path.join(code_dir, '..', 'output',    'archival'))
if os.path.isdir(archival_reduct_base):
    for prog_dir in sorted(glob.glob(os.path.join(archival_reduct_base, '*'))):
        prog_id = os.path.basename(prog_dir)
        for target_path in sorted(glob.glob(os.path.join(prog_dir, '*'))):
            if os.path.isdir(target_path):
                all_targets.append((
                    os.path.join(archival_data_base, prog_id),
                    prog_dir,
                    os.path.join(archival_out_base, prog_id),
                    os.path.basename(target_path),
                ))

for eff_data_dir, eff_reduct_dir, eff_out_dir, target in all_targets:
    os.makedirs(os.path.join(eff_out_dir, target), exist_ok=True)
    target_dir = os.path.join(eff_data_dir, target)
    phot_pars_file = os.path.join(eff_reduct_dir, target, 'phot_pars')
    ref_rootname = None
    if os.path.isfile(phot_pars_file):
        with open(phot_pars_file) as f:
            for line in f:
                if line.startswith('img0_file='):
                    val = line.strip().split('=', 1)[1].strip()
                    ref_rootname = val.rsplit('.chip', 1)[0]
    else:
        global_logger.warning(f'phot_pars not found for {target}. Falling back to filterdrizimg[1] as WCS reference.')
    drizfilelist = glob.glob(os.path.join(target_dir,'*drc.fits'))
    instrument = None
    for imgpath in drizfilelist:
        hdu = fits.open(imgpath)
        header = hdu[0].header
        if 'ACS' in header['INSTRUME']:
            instrument = 'ACS'
        elif 'WFC3' in header['INSTRUME']:
            instrument = 'WFC3'
        else:
            global_logger.warning(f'Instrument not set for {target}. Skipping CMD creation.')
            continue
    if instrument is None:
        global_logger.warning(f'No DRC files found for {target}. Skipping.')
        continue
    dolphot_outfile = os.path.join(eff_reduct_dir, target, f'{target}_{instrument.lower()}')
    _fake_std = os.path.join(eff_reduct_dir, target, f'{target}_{instrument.lower()}.fake')
    _fake_00  = os.path.join(eff_reduct_dir, target, f'{target}_{instrument.lower()}_00.fake')
    ast_file  = _fake_std if os.path.isfile(_fake_std) else _fake_00

    # Determine blue/red filters once per target (needed by both the main
    # catalog block and the AST blocks below, which can run independently
    # of each other), from the .columns file dolphot wrote when it ran.
    red_filter = 'F814W'
    blue_filter = None
    columns_file = dolphot_outfile + '.columns'
    if os.path.isfile(columns_file):
        _, cat_filters, _ = _parse_dolphot_columns(columns_file)
        blue_filter = next((f for f in cat_filters if f != red_filter), None)

    if (os.path.isfile(os.path.join(eff_reduct_dir, target, "dolphot.done")) and
            (not os.path.isfile(os.path.join(eff_out_dir, target, 'phot_target_initial.csv')) or args.overwrite)):

        if blue_filter is None:
            global_logger.warning(f'Could not determine filters for {target} from {columns_file}. Skipping.')
            continue

        dolphot_cat, cat_filters = _read_dolphot_catalog(dolphot_outfile)

        global_logger.info(f"Generating CMDs for {target}.")
        global_logger.info(f"Total dolphot catalog length: {len(dolphot_cat)}")

        global_logger.info(f"Remove sources that are not type 1 or 2 (point-like).")
        condition = (dolphot_cat['type'] < 3)
        dolphot_cat = dolphot_cat[condition]
        global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        global_logger.info(f"Require: mag < {max_mag} (in all filters)")
        condition = (dolphot_cat[f'{blue_filter}_mag'] < max_mag) & (dolphot_cat[f'{red_filter}_mag'] < max_mag)
        dolphot_cat = dolphot_cat[condition]
        global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        global_logger.info(f"Require: crowding < {crowd_thresh} mag")
        condition = (dolphot_cat[f'crowd_{blue_filter}'] + dolphot_cat[f'crowd_{red_filter}'] < crowd_thresh)
        dolphot_cat = dolphot_cat[condition]
        global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        global_logger.info(f"Require: sharpness squared < {max_sharp}")
        condition = ((dolphot_cat[f'sharp_{blue_filter}'] + dolphot_cat[f'sharp_{red_filter}'])**2. < max_sharp)
        dolphot_cat = dolphot_cat[condition]
        global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        
        filters = []
        for imgpath in drizfilelist:
            hdu = fits.open(imgpath)
            header = hdu[0].header
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
        if ref_rootname is not None:
            ref_drc_imgfile = os.path.join(target_dir, ref_rootname+'.fits')
        else:
            global_logger.warning(f'Could not read img0_file from phot_pars for {target}. Falling back to filterdrizimg[1].')
            ref_drc_imgfile = os.path.join(target_dir, filterdrizimg[1]+'.fits')

        # Get reference WCS from DRC image header
        global_logger.info(f'Opening reference WCS from {ref_drc_imgfile}.fits.')
        ref_hdu = fits.open(ref_drc_imgfile)
        ref_WCS = WCS(ref_hdu[1].header,naxis=2)

        #Calculate extinction corrections
        sfd = SFDQuery()
        # Dolphot reports 1-indexed pixel coordinates offset by +0.5 relative to the
        # standard 0-indexed FITS convention; subtract 0.5 to recover the correct position.
        coords = pixel_to_skycoord(numpy.array(dolphot_cat['x'])-0.5, numpy.array(dolphot_cat['y'])-0.5, ref_WCS)
        dolphot_cat['x'] = numpy.array(dolphot_cat['x'])-0.5
        dolphot_cat['y'] = numpy.array(dolphot_cat['y'])-0.5
        dolphot_cat['ra'] = coords.ra.deg
        dolphot_cat['dec'] = coords.dec.deg
        dolphot_cat['E(B-V)'] = sfd(coords)
        dolphot_cat[f'A_{red_filter}'] = dolphot_cat['E(B-V)'] * EXTINCTION_COEFFS[(instrument, red_filter)]
        dolphot_cat[f'A_{blue_filter}'] = dolphot_cat['E(B-V)'] * EXTINCTION_COEFFS[(instrument, blue_filter)]
        dolphot_cat[f'{red_filter}_0'] = dolphot_cat[f'{red_filter}_mag'] - dolphot_cat[f'A_{red_filter}']
        dolphot_cat[f'{blue_filter}_0'] = dolphot_cat[f'{blue_filter}_mag'] - dolphot_cat[f'A_{blue_filter}']
        dolphot_cat['SNR'] = dolphot_cat['SNR_global']

        # Johnson-system (UBVRI) columns, named after whichever letter each
        # filter actually maps to (see JOHNSON_LETTER); skipped with a
        # warning if a filter has no known Johnson extinction coefficient.
        out_cols = ['x', 'y', 'ra', 'dec',
                   f'{blue_filter}_0', f'e_{blue_filter}', f'{red_filter}_0', f'e_{red_filter}']
        for filt in (blue_filter, red_filter):
            letter = JOHNSON_LETTER.get(filt)
            if letter is not None and letter in JOHNSON_EXTINCTION:
                dolphot_cat[f'A_{letter}'] = dolphot_cat['E(B-V)'] * JOHNSON_EXTINCTION[letter]
                dolphot_cat[f'{letter}_0'] = dolphot_cat[f'{filt}_{letter}vega'] - dolphot_cat[f'A_{letter}']
                out_cols.append(f'{letter}_0')
            else:
                global_logger.warning(f'No Johnson extinction coefficient for {filt}; skipping its UBVRI column.')
        out_cols.append('E(B-V)')
        out_cols += [f'A_{blue_filter}', f'A_{red_filter}']
        for filt in (blue_filter, red_filter):
            letter = JOHNSON_LETTER.get(filt)
            if letter is not None and letter in JOHNSON_EXTINCTION:
                out_cols.append(f'A_{letter}')
        out_cols += ['SNR', f'SNR_{blue_filter}', f'SNR_{red_filter}']

        dolphot_cat = dolphot_cat[out_cols]
        phot_outfile = os.path.join(eff_out_dir,target,'phot_full.csv')
        global_logger.info(f'Saving full FoV photometry catalog to {phot_outfile}.')
        dolphot_cat.to_csv(phot_outfile,index=False)


        plt.figure(figsize=(4,8))
        plt.scatter(dolphot_cat[f'{blue_filter}_0']-dolphot_cat[f'{red_filter}_0'], dolphot_cat[f'{red_filter}_0'],c='k',s=3,marker='o')
        plt.ylim(27.5,20)
        plt.xlim(-1,2)
        plt.title(f'{target} Full Field')
        plt.xlabel(f'{blue_filter}$_0$ - {red_filter}$_0$')
        plt.ylabel(f'{red_filter}$_0$')
        plt.savefig(os.path.join(eff_out_dir,target,'CMD_full.pdf'),bbox_inches='tight')
        plt.close()


        # Select the sources within the (circular) 2*r_e of the target
        google_sheet_id = '1MFvVh57tIhzc6vUUmrCvDwyYzSojJTZtpqzXfRgC48s'
        sample_url = f"https://docs.google.com/spreadsheets/d/{google_sheet_id}/export?format=csv"
        hcns_sample = pandas.read_csv(sample_url, skiprows=[1])
        hcns_sample['r_e_arcsec'] = (3600*180/(1E6*numpy.pi))*numpy.where(numpy.isfinite(hcns_sample['D_sat']),
                                                                          hcns_sample['R_e']/hcns_sample['D_sat'],
                                                                          hcns_sample['R_e']/hcns_sample['D_host'])

        hcns_sample['Name'] = hcns_sample['Name'].str.upper()
        hcns_sample = hcns_sample.set_index('Name')

        try:
            target_key = target.upper()
            target_ra, target_dec, target_re = (hcns_sample['RA'][target_key],
                                                hcns_sample['Dec'][target_key],
                                                hcns_sample['r_e_arcsec'][target_key])
    
            target_pos = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg)
    
            dolphot_cat['separation'] = target_pos.separation(coords).arcsec
            dolphot_cat = dolphot_cat[dolphot_cat['separation'] < 2.*target_re]
    
            dolphot_cat = dolphot_cat[out_cols]
            phot_outfile = os.path.join(eff_out_dir,target,'phot_target_initial.csv')
            global_logger.info(f'Saving initial target photometry catalog to {phot_outfile}.')
            dolphot_cat.to_csv(phot_outfile,index=False)

            plt.figure(figsize=(4,8))
            plt.scatter(dolphot_cat[f'{blue_filter}_0']-dolphot_cat[f'{red_filter}_0'], dolphot_cat[f'{red_filter}_0'],c='k',s=3,marker='o')
            plt.ylim(27.5,20)
            plt.xlim(-1,2)
            plt.title(f'{target} (Initial)')
            plt.xlabel(f'{blue_filter}$_0$ - {red_filter}$_0$')
            plt.ylabel(f'{red_filter}$_0$')
            plt.savefig(os.path.join(eff_out_dir,target,'CMD_initial.pdf'),bbox_inches='tight')
            plt.close()
        except:
            global_logger.warning(f'{target} could not be match to HCNS sample table. No target CMD will be produced.')



    if (os.path.isfile(os.path.join(eff_reduct_dir, target, "fakestars.done")) and
            (not os.path.isfile(os.path.join(eff_out_dir, target, 'phot_ast.csv')) or args.overwrite)):

        if blue_filter is None:
            global_logger.warning(f'Could not determine filters for {target} from {columns_file}. Skipping AST.')
            continue

        fake_stars, _ = _read_fake_catalog(ast_file, dolphot_outfile)

        global_logger.info(f'Fake star catalog length for {target}: {len(fake_stars)}')
        global_logger.info(f'Recovery fraction: {numpy.sum(fake_stars["recovered"])/len(fake_stars)}')

        ast_outfile = os.path.join(eff_out_dir,target,'phot_ast.csv')
        global_logger.info(f'Saving AST photometry catalog to {ast_outfile}.')
        fake_stars.to_csv(ast_outfile,index=False)


        # Calculate completeness limits: standard, then a stricter variant
        # requiring SNR >= 4 in both filters for a star to count as recovered.
        _fit_completeness(fake_stars, blue_filter, red_filter, eff_out_dir, target)
        _fit_completeness(fake_stars, blue_filter, red_filter, eff_out_dir, target,
                          suffix='_4sig', min_snr=4)

    elif not os.path.isfile(os.path.join(eff_reduct_dir, target, "dolphot.done")):
        global_logger.info(f'Photometry for {target} incomplete. Skipping.')

    # Check for extra AST iterations produced by --ast mode in HCNS_dolphot.py
    extra_fake_files = sorted(
        f for f in glob.glob(
            os.path.join(eff_reduct_dir, target, f'{target}_{instrument.lower()}_??.fake'))
        if not os.path.basename(f).endswith('_00.fake'))
    ast_full_path = os.path.join(eff_out_dir, target, 'phot_ast_full.csv')
    _ast_full_mtime = os.path.getmtime(ast_full_path) if os.path.isfile(ast_full_path) else 0
    _new_extra = any(os.path.getmtime(ef) > _ast_full_mtime for ef in extra_fake_files)
    if (extra_fake_files and
            os.path.isfile(os.path.join(eff_out_dir, target, 'phot_ast.csv')) and
            (_new_extra or not os.path.isfile(ast_full_path) or args.overwrite)):
        if blue_filter is None:
            global_logger.warning(f'Could not determine filters for {target} from {columns_file}. Skipping extra ASTs.')
            continue
        global_logger.info(
            f'{len(extra_fake_files)} extra AST file(s) found for {target}. Building full catalog.')
        dfs = [pandas.read_csv(os.path.join(eff_out_dir, target, 'phot_ast.csv'))]
        for ef in extra_fake_files:
            extra_df, _ = _read_fake_catalog(ef, dolphot_outfile)
            dfs.append(extra_df)
        fake_stars_full = pandas.concat(dfs, ignore_index=True)
        global_logger.info(f'Full AST catalog for {target}: {len(fake_stars_full)} stars.')
        fake_stars_full.to_csv(ast_full_path, index=False)

        # Re-fit completeness curves on the full catalog, overwriting
        # completeness{,_4sig}.dat/pdf
        _fit_completeness(fake_stars_full, blue_filter, red_filter, eff_out_dir, target,
                          log_suffix=' (full AST catalog)')
        _fit_completeness(fake_stars_full, blue_filter, red_filter, eff_out_dir, target,
                          suffix='_4sig', min_snr=4, log_suffix=' (full AST catalog)')





