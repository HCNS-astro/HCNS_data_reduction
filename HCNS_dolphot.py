import os, sys, glob, numpy
import shutil, subprocess, logging
from astropy.io import fits
from joblib import Parallel, delayed

code_dir = os.getcwd()
data_dir = os.path.abspath(os.path.join(code_dir,'..','data'))
reduct_dir = os.path.abspath(os.path.join(code_dir,'..','reduction'))


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

def close_logger(logger_instance):
    handlers = logger_instance.handlers[:]
    for handler in handlers:
        handler.close()
        logger_instance.removeHandler(handler)

def execute_command(command, exec_dir, logger):

    logger.info(f'Running command: {" ".join(command)}')

    with subprocess.Popen(command, cwd=exec_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1) as process:
        for line in process.stdout:
            logger.info(line.strip())
        process.wait()
    return process.returncode

def align_dolphot(target, logger, instrument=None):

    dolphot_dir = os.path.join(reduct_dir,target)

    if 'WFC3' in instrument:
        command = ["dolphot", f"{target}_wfc3", "-pphot_pars"]
    elif 'ACS' in instrument:
        command = ["dolphot", f"{target}_acs", "-pphot_pars"]
    else:
        logger.warning(f'Instrument not set correctly. Must be either "ACS" or "WFC3", but is {instrument}.')
        return False

    result = subprocess.run(command, cwd=dolphot_dir, text=True, check=True, 
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    sig_values = []
    n_stars_align = []
    for line in result.stdout.splitlines():
        logger.info(line)
        if "sig" in line:
            inx1 = line.index("sig")
            if "rsig" in line:
                inx2 = line.index(", rsig")
            else:
                inx2 = len(line)
            sig_values.append(float(line[inx1+4:inx2]))
            inx1 = line.find("matched")
            inx2 = line.find(" used")
            n_stars_align.append(int(line[inx1+9:inx2]))
        sys.stdout.flush()

    if numpy.any(numpy.array(sig_values) > 0.75) or numpy.any(numpy.array(n_stars_align) < 15):
        #Alignment failed
        return False
    else:
        #Alignment succeeded
        return True

def prep_dolphot(target, CTE=False, align_iter=5, verbose=False, template_file=None):

    if verbose:
        global_logger.info(f'Prepping files for running dolphot on {target}.')

    target_dir = os.path.join(data_dir,target)
    dolphot_dir = os.path.join(reduct_dir,target)

    #Create dir for running dolphot
    if os.path.isdir(dolphot_dir):
        if verbose:
            global_logger.info(f'{dolphot_dir} already exists.')
    else:
        os.mkdir(dolphot_dir)
        global_logger.info(f'{dolphot_dir} created.')

    #Create log for dolphot run for this target
    dolphot_logger = make_logger("dolphot", filename=os.path.join(dolphot_dir, f"{target}_dolphot.log"))

    #Find all raw files that were downloaded
    if CTE:
        raw_expfilelist = glob.glob(os.path.join(target_dir,'*flt.fits'))
    else:
        raw_expfilelist = glob.glob(os.path.join(target_dir,'*flc.fits'))
    raw_drizfilelist = glob.glob(os.path.join(target_dir,'*drc.fits'))

    #Copy files to dolphot dir
    expfilelist = []
    drizfilelist = []
    for imgpath in raw_expfilelist:
        imgfile = os.path.split(imgpath)[1]
        copypath = os.path.join(dolphot_dir,imgfile)
        if not os.path.isfile(copypath):
            global_logger.info(f'Copying {imgfile} to {copypath}')
            shutil.copyfile(imgpath,copypath)
        elif verbose:
            global_logger.info(f'{copypath} already exists.')
        expfilelist.append(copypath)
    for imgpath in raw_drizfilelist:
        imgfile = os.path.split(imgpath)[1]
        copypath = os.path.join(dolphot_dir,imgfile)
        if not os.path.isfile(copypath):
            global_logger.info(f'Copying {imgfile} to {copypath}')
            shutil.copyfile(imgpath,copypath)
        elif verbose:
            global_logger.info(f'{copypath} already exists.')
        drizfilelist.append(copypath)

    #Extract info from files headers
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
            global_logger.warning(f'Instrument not set for {target}. Dolphot prep failed for {target}.')
            return None
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
    filters = list(set(filters))
    filters.sort()
    if verbose:
        global_logger.info(f'Filters found for {target}: {", ".join(filters)}')

    filterdrizimg = []
    filterexposures = []
    filterexptimes = []

    for i,filtername in enumerate(filters):
        filterexposures.append([])
        filterexptimes.append([])
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
        for imgpath in expfilelist:
            imgfile = os.path.split(imgpath)[1]
            inx = imgfile.find('.fits')
            rootname = imgfile[:inx]
            hdu = fits.open(imgpath)
            header = hdu[0].header
            match instrument:
                case 'WFC3':
                    if filtername in header['FILTER']:
                        filterexposures[i].append(rootname)
                        filterexptimes[i].append(header['EXPTIME'])
                case 'ACS':
                    if filtername in header['FILTER1']:
                        filterexposures[i].append(rootname)
                        filterexptimes[i].append(header['EXPTIME'])
                    elif filtername in header['FILTER2']:
                        filterexposures[i].append(rootname)
                        filterexptimes[i].append(header['EXPTIME'])

    #Make photometry parameters file
    if not os.path.isfile(os.path.join(dolphot_dir, "phot_pars")):
        photpars = open(os.path.join(dolphot_dir,'phot_pars'),'w')
        photpars.write(f"Nimg={2*len(numpy.array(filterexposures).flatten())}\n")
        filterwavelengths = []
        for i,filtername in enumerate(filters):
            filterwavelengths.append(int(filtername[1:4]))
        inx = filterwavelengths.index(min(filterwavelengths))
        photpars.write(f"img0_file={filterdrizimg[inx]}.chip1\n")
        for i,filename in enumerate(numpy.array(filterexposures).flatten()):
            photpars.write("img{0}_file={1}.chip1\nimg{0}_shift= 0 0\nimg{0}_xform= 1 0 0\n".format(2*i+1,filename))
            photpars.write("img{0}_file={1}.chip2\nimg{0}_shift= 0 0\nimg{0}_xform= 1 0 0\n".format(2*i+2,filename))

        if template_file is None:
            match instrument:
                case 'WFC3': template_file = 'phot_pars_WFC3_template'
                case 'ACS': template_file = 'phot_pars_ACS_template'
        temp_photpars = open(template_file)

        lines = temp_photpars.readlines()
        temp_photpars.close()
        for line in lines:
            photpars.write(line)
        photpars.write('\n')

        if CTE:
            photpars.write(f"{instrument.upper()}useCTE = 1\n")
        else:
            photpars.write(f"{instrument.upper()}useCTE = 0\n")
        photpars.write(f"AlignIter={align_iter}\n")
        photpars.write("Align = 2\nRotate = 1\nUseWCS = 1\nAlignOnly = 1\n")
        photpars.close() 
        global_logger.info(f'Created photpars file for {target}.')

    #Run masking
    if not os.path.isfile(os.path.join(dolphot_dir, f"{instrument.lower()}mask.done")):
        dolphot_logger.info(f'Running masking for {target}.')
        for i,filtername in enumerate(filters):
            command = [f"{instrument.lower()}mask", f"-exptime={numpy.sum(filterexptimes[i])}", f"-ncombine={len(filterexptimes[i])}", f"{filterdrizimg[i]}.fits"]
            execute_command(command, dolphot_dir, dolphot_logger)

            for j,filename in enumerate(filterexposures[i]):
                command = [f"{instrument.lower()}mask", f"{filename}.fits"]
                execute_command(command, dolphot_dir, dolphot_logger)
        subprocess.run(["touch", f"{instrument.lower()}mask.done"], cwd=dolphot_dir)

    #Run split groups
    if not os.path.isfile(os.path.join(dolphot_dir, "splitgroups.done")):
        dolphot_logger.info(f'Running splitgroups for {target}.')
        for i,filtername in enumerate(filters):
            command = ["splitgroups", f"{filterdrizimg[i]}.fits"]
            execute_command(command, dolphot_dir, dolphot_logger)

            for j,filename in enumerate(filterexposures[i]):
                command = ["splitgroups", f"{filename}.fits"]
                execute_command(command, dolphot_dir, dolphot_logger)
        subprocess.run(["touch", "splitgroups.done"], cwd=dolphot_dir)

    #Run calc sky
    if not os.path.isfile(os.path.join(dolphot_dir, "calcsky.done")):
        dolphot_logger.info(f'Running calcsky for {target}.')
        for i,filtername in enumerate(filters):
            command = ["calcsky", f"{filterdrizimg[i]}.chip1", "15", "35", "4", "2.25", "2.00"]
            execute_command(command, dolphot_dir, dolphot_logger)
            for j,filename in enumerate(filterexposures[i]):
                for chip in range(1,2):
                    command = ["calcsky", f"{filename}.chip{chip}", "15", "35", "4", "2.25", "2.00"]
                    execute_command(command, dolphot_dir, dolphot_logger)
        subprocess.run(["touch", "calcsky.done"], cwd=dolphot_dir)

    #Run alignment
    if (not os.path.isfile(os.path.join(dolphot_dir, "align.done"))) and (not os.path.isfile(os.path.join(dolphot_dir, "align.failed"))):
        dolphot_logger.info(f'Running alignment for {target}.')
        align_success = False
        align_attempt = 0
        while not align_success:
            align_success = align_dolphot(target, dolphot_logger, instrument=instrument)
            align_attempt += 1
            if not align_success:
                dolphot_logger.info(f'Alignment attempt {align_attempt} failed.')
                if align_attempt < 3:
                    dolphot_logger.info('Re-running alignment.')
                    temp_photpars = open(os.path.join(dolphot_dir,'phot_pars'))
                    lines = temp_photpars.readlines()
                    temp_photpars.close()
                    photpars = open(os.path.join(dolphot_dir,'phot_pars'), 'w')
                    for line in lines:
                        if 'Align = ' not in line:
                            photpars.write(line)
                        else:
                            break
                    match align_attempt:
                        case 1:
                            photpars.write("Align = 3\nRotate = 1\nUseWCS = 1\nAlignOnly = 1\n")
                            photpars.close() 
                        case 2:
                            photpars.write("Align = 4\nRotate = 1\nUseWCS = 1\nAlignOnly = 1\n")
                            photpars.close() 
                    dolphot_logger.info('Updated alignment parameters in phot_pars.')
                elif align_attempt == 3:
                    dolphot_logger.info(f'Re-running alignment (final attempt).')
                    temp_photpars = open(os.path.join(dolphot_dir,'phot_pars'))
                    lines = temp_photpars.readlines()
                    temp_photpars.close()
                    photpars = open(os.path.join(dolphot_dir,'phot_pars'), 'w')
                    for line in lines:
                        if 'img0_file' in line:
                            #Switch which image is used as reference
                            inx = filterwavelengths.index(max(filterwavelengths))
                            photpars.write(f"img0_file={filterdrizimg[inx]}.chip1\n")
                        elif 'Align = ' not in line:
                            photpars.write(line)
                        else:
                            break
                    photpars.write("Align = 3\nRotate = 1\nUseWCS = 1\nAlignOnly = 1\n")
                    photpars.close()
                else:
                    break

        if align_success:
            subprocess.run(["touch", "align.done"], cwd=dolphot_dir)
            dolphot_logger.info(f'Alignment completed successfully for {target}.')
            global_logger.info(f'Alignment completed successfully for {target}.')
        else:
            subprocess.run(["touch", "align.failed"], cwd=dolphot_dir)
            dolphot_logger.warning(f'Alignment checks failed for {target}. Manual alignment may be needed.')
            global_logger.warning(f'Alignment checks failed for {target}. Manual alignment may be needed.')

    #Close logger
    handlers = dolphot_logger.handlers[:] 
    for handler in handlers:
        handler.close()  
        dolphot_logger.removeHandler(handler)

    return align_success

def generate_fake_stars(target, dolphot_logger, Nfake=200000, 
                        filt_min=20, filt_max=30.5, col_min=-1.0, col_max=2.5):

    dolphot_logger.info(f'Generating fake stars for {target}.')

    dolphot_dir = os.path.join(reduct_dir,target)

    drizfilelist = glob.glob(os.path.join(dolphot_dir,'*drc.fits'))

    #Extract info from files headers
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
            global_logger.warning(f'Instrument not set. Fake stars failed for {target}.')
            return None
        filtername = header['FILTER']
        filters.append(filtername)
    filters = list(set(filters))
    filters.sort()

    if len(filters) > 1:
        global_logger.info(f'Fake stars will be generated for filters: {filters[0]} and {filters[-1]}.')
        command = ['fakelist', f'{target}_{instrument.lower()}', f'{instrument.upper()}_{filters[0]}', f'{instrument.upper()}_{filters[-1]}',
                   f'{filt_min}', f'{filt_max}', f'{col_min}', f'{col_max}', f'-nstar={Nfake}']
        #execute_command(command, dolphot_dir, dolphot_logger)
        with open(os.path.join(dolphot_dir,'fakelist.dat'), "w") as f:
            subprocess.run(command, cwd=dolphot_dir, stdout=f)
    else:
        global_logger.error(f'There are not enough filters to generate fake stars.')

    return None

def run_dolphot(target, fake_stars=False, verbose=False):

    dolphot_dir = os.path.join(reduct_dir,target)

    #Open log for dolphot run for this target
    dolphot_logger = make_logger("dolphot", filename=os.path.join(dolphot_dir, f"{target}_dolphot.log"))

    #Remove align only parameter
    with open(os.path.join(dolphot_dir,'phot_pars'), 'r') as file:
        lines = file.readlines()
    with open(os.path.join(dolphot_dir,'phot_pars'), 'w') as file:
        for line in lines:
            if "AlignOnly" not in line:
                file.write(line)

    if fake_stars:
        with open(os.path.join(dolphot_dir,'phot_pars'), 'r') as file:
            lines = file.readlines()
        with open(os.path.join(dolphot_dir,'phot_pars_fake'), 'w') as file:
            for line in lines:
                file.write(line)
            file.write('FakeStars = fakelist.dat\nRandomFake=1')

        generate_fake_stars(target, dolphot_logger)

    #Set instrument
    instrument = None
    drizfilelist = glob.glob(os.path.join(dolphot_dir,'*drc.fits'))
    for imgpath in drizfilelist:
        hdu = fits.open(imgpath)
        header = hdu[0].header
        if 'ACS' in header['INSTRUME']:
            instrument = 'ACS'
        elif 'WFC3' in header['INSTRUME']:
            instrument = 'WFC3'
        else:
            global_logger.warning(f'Instrument not set. Dolphot failed for {target}.')
            return None

    #Run dolphot
    dolphot_logger.info(f'Running dolphot for {target}.')
    if not fake_stars:
        command = ["dolphot", f"{target}_wfc3", "-pphot_pars"]
    else:
        command = ["dolphot", f"{target}_wfc3", "-pphot_pars_fake"]
    execute_command(command, dolphot_dir, dolphot_logger)
    dolphot_logger.info(f'Dolphot completed for {target}.')
    global_logger.info(f'Dolphot completed for {target}.')
    if not fake_stars:
        subprocess.run(["touch", "dolphot.done"], cwd=dolphot_dir)
    else:
        subprocess.run(["touch", "fakestars.done"], cwd=dolphot_dir)

    #Close logger
    handlers = dolphot_logger.handlers[:] 
    for handler in handlers:
        handler.close()  
        dolphot_logger.removeHandler(handler)

    return None
    


def _prep_dolphot(path):
    """Worker function for parallel dolphot prep execution."""
    target = str(os.path.split(path)[1])

    if os.path.isfile(os.path.join(reduct_dir, target, "align.failed")):
        global_logger.warning(f'Dolphot prep for {target} already ran, but alignment failed. Skipping.')
    elif not os.path.isfile(os.path.join(reduct_dir, target, "align.done")):
        try:
            tmp = prep_dolphot(target, CTE=CTE)
        except:
            global_logger.warning(f'Dolphot prep failed for {target}.')


def _run_dolphot_if_ready(path, reduct_dir, fake_stars=False):
    """Worker function for parallel dolphot execution."""
    target = str(os.path.split(path)[1])

    if fake_stars:
        if (os.path.isfile(os.path.join(reduct_dir, target, "dolphot.done")) and
            not os.path.isfile(os.path.join(reduct_dir, target, "fakestars.done"))):
            tmp = run_dolphot_WFC3(target, fake_stars=True)
    else:
        if not os.path.isfile(os.path.join(reduct_dir, target, "align.done")):
            global_logger.warning(f'Alignment incomplete for {target}. Skipping.')
        elif not os.path.isfile(os.path.join(reduct_dir, target, "dolphot.done")):
            tmp = run_dolphot_WFC3(target)




#Execute functions
global_logger = make_logger("global", filename="HCNS_dolphot.log")

N_CPU = 3
CTE = True

#Prep dolphot
global_logger.info(f'Running dolphot prep for all downloaded targets.')
paths = glob.glob(os.path.join(data_dir, "*"))

Parallel(n_jobs=N_CPU)(
    delayed(_prep_dolphot)(path)
    for path in paths
)

#Run dolphot
global_logger.info(f'Running dolphot for all downloaded targets.')

Parallel(n_jobs=N_CPU)(
    delayed(_run_dolphot_if_ready)(path, reduct_dir, fake_stars=False)
    for path in paths
)

#Run ASTs
global_logger.info(f'Running ASTs for all available targets.')

Parallel(n_jobs=N_CPU)(
    delayed(_run_dolphot_if_ready)(path, reduct_dir, fake_stars=True)
    for path in paths
)