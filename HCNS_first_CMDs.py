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
R_ACS_F814W =  1.526 # ACS values from Schlafly and Finkbeiner (2011)
R_ACS_F606W = 2.471 
R_I = 1.505 # Landolt values from Schlafly and Finkbeiner (2011)
R_V = 2.742

max_mag = 30.
max_sharp = 0.1
crowd_thresh = 1.0

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
    
    dolphot_outfile = os.path.join(eff_reduct_dir, target, f'{target}_{instrument.lower()}')
    ast_file = os.path.join(eff_reduct_dir, target, f'{target}_{instrument.lower()}.fake')

    if (os.path.isfile(os.path.join(eff_reduct_dir, target, "dolphot.done"))):
        #and not os.path.isfile(os.path.join(eff_out_dir,target,'phot_target_initial.csv'))):

        dolphot_cat = pandas.read_csv(dolphot_outfile, sep=r'\s+', header=None)

        # All column IDs are hard-coded for WFC3 with two filters
        global_logger.info(f"Generating CMDs for {target}.")
        global_logger.info(f"Total dolphot catalog length: {len(dolphot_cat)}")

        global_logger.info(f"Remove sources that are not type 1 or 2 (point-like).")
        condition = (dolphot_cat[10] < 3)
        dolphot_cat = dolphot_cat[condition]
        global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        #global_logger.info(f"Remove sources with bad photometry flags.")
        #condition = (dolphot_cat[23] < 4) & (dolphot_cat[36] < 4)
        #dolphot_cat = dolphot_cat[condition]
        #global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        global_logger.info(f"Require: mag < {max_mag} (in all filters)")
        condition = (dolphot_cat[15] < max_mag) & (dolphot_cat[28] < max_mag)
        dolphot_cat = dolphot_cat[condition]
        global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        global_logger.info(f"Require: crowding < {crowd_thresh} mag")
        condition = (dolphot_cat[22] + dolphot_cat[35] < crowd_thresh)
        dolphot_cat = dolphot_cat[condition]
        global_logger.info(f"New catalogue length: {len(dolphot_cat)}")

        global_logger.info(f"Require: sharpness squared < {max_sharp}")
        condition = ((dolphot_cat[20] + dolphot_cat[33])**2. < max_sharp)
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
        # Assume that the reference image is F814W 
        # THIS NEEDS TO BE FIXED
        ref_drc_imgfile = os.path.join(target_dir, filterdrizimg[1]+'.fits')

        # Get reference WCS from DRC image header
        global_logger.info(f'Opening reference WCS from {ref_drc_imgfile}.fits.')
        ref_hdu = fits.open(ref_drc_imgfile)
        ref_WCS = WCS(ref_hdu[1].header,naxis=2)

        #Calculate extinction corrections
        sfd = SFDQuery()
        # Dolphot reports 1-indexed pixel coordinates offset by +0.5 relative to the
        # standard 0-indexed FITS convention; subtract 0.5 to recover the correct position.
        coords = pixel_to_skycoord(numpy.array(dolphot_cat[2])-0.5, numpy.array(dolphot_cat[3])-0.5, ref_WCS)
        dolphot_cat['x'] = numpy.array(dolphot_cat[2])-0.5
        dolphot_cat['y'] = numpy.array(dolphot_cat[3])-0.5
        dolphot_cat['ra'] = coords.ra.deg
        dolphot_cat['dec'] = coords.dec.deg
        dolphot_cat['E(B-V)'] = sfd(coords)
        match instrument:
            case 'ACS':
                dolphot_cat['A_F814W'] = dolphot_cat['E(B-V)']*R_ACS_F814W
                dolphot_cat['A_F606W'] = dolphot_cat['E(B-V)']*R_ACS_F606W
            case 'WFC3':
                dolphot_cat['A_F814W'] = dolphot_cat['E(B-V)']*R_WFC3_F814W
                dolphot_cat['A_F606W'] = dolphot_cat['E(B-V)']*R_WFC3_F606W
        dolphot_cat['A_I'] = dolphot_cat['E(B-V)']*R_I
        dolphot_cat['A_V'] = dolphot_cat['E(B-V)']*R_V
        dolphot_cat['F814W_0'] = dolphot_cat[28] - dolphot_cat['A_F814W']
        dolphot_cat['F606W_0'] = dolphot_cat[15] - dolphot_cat['A_F606W']
        dolphot_cat['I_0'] = dolphot_cat[29] - dolphot_cat['A_I']
        dolphot_cat['V_0'] = dolphot_cat[16] - dolphot_cat['A_V']
        dolphot_cat['e_F814W'] = dolphot_cat[30]
        dolphot_cat['e_F606W'] = dolphot_cat[17]
        dolphot_cat['SNR'] = dolphot_cat[5]
        dolphot_cat['SNR_F606W'] = dolphot_cat[19]
        dolphot_cat['SNR_F814W'] = dolphot_cat[32]

        dolphot_cat = dolphot_cat[['x','y','ra','dec','F606W_0','e_F606W','F814W_0','e_F814W',
                                   'V_0','I_0','E(B-V)','A_F606W','A_F814W','A_V','A_I',
                                   'SNR','SNR_F606W','SNR_F814W']]
        phot_outfile = os.path.join(eff_out_dir,target,'phot_full.csv')
        global_logger.info(f'Saving full FoV photometry catalog to {phot_outfile}.')
        dolphot_cat.to_csv(phot_outfile,index=False)


        plt.figure(figsize=(4,8))
        plt.scatter(dolphot_cat['F606W_0']-dolphot_cat['F814W_0'], dolphot_cat['F814W_0'],c='k',s=3,marker='o')
        plt.ylim(27.5,20)
        plt.xlim(-0.5,1.5)
        plt.title(f'{target} Full Field')
        plt.xlabel(r'F606W$_0$ - F814W$_0$')
        plt.ylabel('F814W$_0$')
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
            target_ra, target_dec, target_re = hcns_sample['RA'][target],hcns_sample['Dec'][target],hcns_sample['r_e_arcsec'][target]
    
            target_pos = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg)
    
            dolphot_cat['separation'] = target_pos.separation(coords).arcsec
            dolphot_cat = dolphot_cat[dolphot_cat['separation'] < 2.*target_re]
    
            dolphot_cat = dolphot_cat[['x','y','ra','dec','F606W_0','e_F606W','F814W_0','e_F814W',
                                       'V_0','I_0','E(B-V)','A_F606W','A_F814W','A_V','A_I',
                                       'SNR','SNR_F606W','SNR_F814W']]
            phot_outfile = os.path.join(eff_out_dir,target,'phot_target_initial.csv')
            global_logger.info(f'Saving initial target photometry catalog to {phot_outfile}.')
            dolphot_cat.to_csv(phot_outfile,index=False)
    
            plt.figure(figsize=(4,8))
            plt.scatter(dolphot_cat['F606W_0']-dolphot_cat['F814W_0'], dolphot_cat['F814W_0'],c='k',s=3,marker='o')
            plt.ylim(27.5,20)
            plt.xlim(-0.5,1.5)
            plt.title(f'{target} (Initial)')
            plt.xlabel(r'F606W$_0$ - F814W$_0$')
            plt.ylabel('F814W$_0$')
            plt.savefig(os.path.join(eff_out_dir,target,'CMD_initial.pdf'),bbox_inches='tight')
            plt.close()
        except:
            global_logger.warning(f'{target} could not be match to HCNS sample table. No target CMD will be produced.')



    if (os.path.isfile(os.path.join(eff_reduct_dir, target, "fakestars.done"))): 
        #and not os.path.isfile(os.path.join(eff_out_dir,target,'phot_ast.csv'))):

        # Now create file for fake stars
        fake_stars = pandas.read_csv(ast_file, sep=r'\s+', header=None)

        # TODO: Nimages is hard-coded for 4 exposures × 2 chips. The column offsets
        # c1–c4 below depend on this value and will be wrong if the number of images changes.
        Nimages = 8
        # Column offsets derived from the dolphot fake-star output format:
        # c1: start of the second filter's global photometry block
        # c2: start of the per-source summary columns (chi, snr, sharp, etc.)
        # c3/c4: per-filter individual photometry columns for filter 1 and filter 2
        c1 = 5 + Nimages
        c2 = c1 + Nimages + 3
        c3 = c2 + 6 + 5
        #For later versions of dolphot the sharpness, roundness, pa, crowding and type indices
        #need to be increased by 1
        c4 = c3 + 8 + 5
        fake_stars = fake_stars.rename(columns={2:'x', 3:'y', 5:'F606W_in', c1:'F814W_in', c2:'chi', c2+1:'SNR',
                                                c2+2:'sharpness', c2+3:'roundness', c2+4:'pa', c2+5:'crowding', c2+6:'type',
                                                c3:'F606W_out', c3+1:'V_out', c3+2:'err_F606W_out', c3+3:'chi_F606W',
                                                c3+4:'SNR_F606W', c3+5:'sharpness_F606W', c3+6:'roundness_F606W',
                                                c3+7:'crowding_F606W', c3+8:'flag_F606W',
                                                c4:'F814W_out', c4+1:'I_out', c4+2:'err_F814W_out', c4+3:'chi_F814W',
                                                c4+4:'SNR_F814W', c4+5:'sharpness_F814W', c4+6:'roundness_F814W',
                                                c4+7:'crowding_F814W', c4+8:'flag_F814W'})

        fake_stars['F606W-F814W'] = fake_stars['F606W_in']-fake_stars['F814W_in']

        condition = ((fake_stars['F606W_out'] < 99.) & (fake_stars['F814W_out'] < 99.) & 
                     (fake_stars['type'] < 3) &
                     (fake_stars['crowding_F606W'] + fake_stars['crowding_F814W'] < crowd_thresh) &
                     ((fake_stars['sharpness_F606W'] + fake_stars['sharpness_F814W'])**2. < max_sharp))

        fake_stars['recovered'] = numpy.where(condition,1,0)

        global_logger.info(f'Fake star catalog length for {target}: {len(fake_stars)}')
        global_logger.info(f'Recovery fraction: {numpy.sum(fake_stars['recovered'])/len(fake_stars)}')

        fake_stars['x'] = numpy.array(fake_stars['x'])-0.5
        fake_stars['y'] = numpy.array(fake_stars['y'])-0.5
        fake_stars = fake_stars[['x','y','F606W_in','F814W_in','F606W_out','F814W_out',
                                 'SNR','SNR_F606W','SNR_F814W','recovered']]
        ast_outfile = os.path.join(eff_out_dir,target,'phot_ast.csv')
        global_logger.info(f'Saving AST photometry catalog to {ast_outfile}.')
        fake_stars.to_csv(ast_outfile,index=False)


        # Calculate completeness limits
        m_min = 21.
        m_max = 30.
        m_wid = 0.1

        m_bins = numpy.arange(m_min,m_max+0.1*m_wid,m_wid)

        colmax = 2.0
        colmin = -1.0
        colwid = 0.2
        colbins = numpy.arange(colmin,colmax+0.1*colwid,colwid)

        C90 = numpy.zeros(len(colbins)-1)
        C50 = numpy.zeros(len(colbins)-1)

        global_logger.info(f'Fitting completeness limits for {target}.')
        fake_stars['F606W-F814W'] = fake_stars['F606W_in']-fake_stars['F814W_in']
        for c in range(len(colbins)-1):
            condition = ((fake_stars['F606W-F814W'] < colbins[c+1]) & (fake_stars['F606W-F814W'] > colbins[c]))
            fake_stars_colbin = fake_stars[condition]

            comp = numpy.zeros(len(m_bins)-1)
            cnts = numpy.zeros(len(m_bins)-1)

            for i in fake_stars_colbin.index:
                j = int(max(min(len(m_bins)-2,numpy.floor((fake_stars_colbin['F814W_in'][i]-m_min)/m_wid)),0))
                if fake_stars_colbin['recovered'][i] > 0:
                    comp[j] += 1.
                cnts[j] += 1.

            comp = comp/cnts

            inx = numpy.where(comp > 0.4)[0]
            
            try:
                erf_fit = scipy.optimize.curve_fit(comp_func,m_bins[inx]+0.05,comp[inx],p0=[26.5,0.5],
                                                   sigma=1./numpy.sqrt(cnts[inx]),bounds=[[24.,0.1],[28.,3.]])

                comp90 = inv_comp_func(0.9,erf_fit[0][0],erf_fit[0][1])

                comp50 = erf_fit[0][0]

                C90[c] = comp90
                C50[c] = comp50
            except ValueError:
                C90[c] = numpy.nan
                C50[c] = numpy.nan

        try:        
            fit90 = scipy.optimize.curve_fit(col_comp_func,colbins[:-1]+0.5*colwid,C90,p0=[0.8,26.5,0.])
            global_logger.info(f"90% Completeness parameters: [{fit90[0][0]}, {fit90[0][1]}, {fit90[0][2]}]")

            # 50% completeness varies nearly linearly with colour over this range, so a
            # simple linear model is used rather than the piecewise model applied to 90%.
            fit50 = scipy.optimize.curve_fit(lambda x,a,b: a*x + b,colbins[:-1]+0.5*colwid,C50,p0=[1,30])
            global_logger.info(f"50% Completeness parameters: [{fit50[0][0]}, {fit50[0][1]}]")

            with open(os.path.join(eff_out_dir,target,'completeness.dat'), 'w') as f:
                f.write(f'comp50 = [{fit50[0][0]}, {fit50[0][1]}]\n')
                f.write(f'comp90 = [{fit90[0][0]}, {fit90[0][1]}, {fit90[0][2]}]')

            x_tmp = numpy.arange(-1.,2.0,0.01)
            plt.plot(x_tmp,col_comp_func(x_tmp,fit90[0][0],fit90[0][1],fit90[0][2]))
            plt.plot(x_tmp,fit50[0][0]*x_tmp + fit50[0][1])
        except ValueError:
            global_logger.warning(f'Completeness limit fit failed for {target}.')
        
        plt.scatter(colbins[:-1]+0.5*colwid,C90)
        plt.scatter(colbins[:-1]+0.5*colwid,C50)
        plt.ylim(29,24)
        plt.ylabel('F814W')
        plt.xlabel('F606W-F814W')
        plt.savefig(os.path.join(eff_out_dir,target,'completeness.pdf'),bbox_inches='tight')
        plt.close()

    elif not os.path.isfile(os.path.join(eff_reduct_dir, target, "dolphot.done")):
        global_logger.info(f'Photometry for {target} incomplete. Skipping.')
    else:
        continue





