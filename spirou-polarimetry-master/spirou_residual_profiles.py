# -*- coding: iso-8859-1 -*-
"""
    Created on 15 Dec 2022
    
    Description: This routine calculate the median of a series of LSD profiles and save the median-subtracted "residual" data
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python /Volumes/Samsung_T5/spirou-tools/spirou-polarimetry/spirou_residual_profiles.py --input=2*_lsd.fits

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import numpy as np
import glob

import matplotlib.pyplot as plt
import astropy.io.fits as fits

from scipy.interpolate import interp1d
from scipy import ndimage

import spirouPolarUtils as spu


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input LSD data pattern",type='string',default="*_lsd.fits")
parser.add_option("-r", "--source_rv", dest="source_rv", help="Source radial velocity in km/s",type='float',default=0.)
parser.add_option("-s", action="store_true", dest="star_frame", help="LSD profiles in the star frame", default=False)
parser.add_option("-m", action="store_true", dest="mediancombine", help="mediancombine", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with spirou_residual_profiles.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input LSD data pattern: ', options.input)
    print('Source radial velocity in km/s: ', options.source_rv)

# make list of data files
if options.verbose:
    print("Creating list of lsd files...")
inputdata = sorted(glob.glob(options.input))
#---

bjd = []
lsd_vels = []
lsd_pol, lsd_null, lsd_flux, lsd_fluxmodel  = [], [], [], []
lsd_pol_err, lsd_null_err, lsd_flux_err  = [], [], []

source_rvs = []

for i in range(len(inputdata)) :
    print("Loading LSD profile in file {0}/{1}: {2}".format(i, len(inputdata), inputdata[i]))
    hdu = fits.open(inputdata[i])
    hdr = hdu[0].header + hdu[1].header

    source_rv = np.nan
    fluxmodel = np.full_like(hdu['STOKESI'].data,np.nan)
    try :
        stokesI_fit = spu.fit_lsd_flux_profile(hdu['VELOCITY'].data, hdu['STOKESI'].data, hdu['STOKESI_ERR'].data, guess=None, func_type="gaussian", plot=False)
        fluxmodel = stokesI_fit["MODEL"]
        source_rv = stokesI_fit["VSHIFT"]
    except :
        print("WARNING: Could not fit gaussian to Stokes I profile, skipping file {0}: {2}".format(i, inputdata[i]))
        continue
    
    if "MEANBJD" in hdr.keys() :
        bjd.append(float(hdr["MEANBJD"]))
    elif "BJD" in hdr.keys() :
        bjd.append(float(hdr["BJD"]))
    else :
        print("Could not read BJD from header, exit ...")
        exit()

    if options.source_rv != 0. :
        source_rv = options.source_rv

    source_rvs.append(source_rv)
    
    lsd_vels.append(hdu['VELOCITY'].data)
    lsd_pol.append(hdu['STOKESVQU'].data)
    lsd_null.append(hdu['NULL'].data)
    lsd_flux.append(hdu['STOKESI'].data)
    lsd_fluxmodel.append(fluxmodel)
    
    lsd_pol_err.append(hdu['STOKESVQU_ERR'].data)
    lsd_null_err.append(hdu['NULL_ERR'].data)
    lsd_flux_err.append(hdu['STOKESI_ERR'].data)

    hdu.close()

bjd = np.array(bjd)
source_rvs = np.array(source_rvs)
source_rv = np.nanmedian(source_rvs)

# replace source rv with null values by the median rv
failed = np.isnan(source_rvs)
source_rvs[failed] = np.full_like(source_rvs[failed],source_rv)

vel_min, vel_max = -1e30, +1e30
dvel = []
for i in range(len(inputdata)) :
    min_rv = np.nanmin(lsd_vels[i] - source_rvs[i])
    max_rv = np.nanmax(lsd_vels[i] - source_rvs[i])
    
    if min_rv > vel_min :
        vel_min = min_rv
    if max_rv < vel_max :
        vel_max = max_rv

    dvel.append(np.nanmedian(np.abs(lsd_vels[i][1:]-lsd_vels[i][:-1])))

dvel = np.array(dvel)

# create an output velocity array
out_vels = np.arange(vel_min, vel_max, np.nanmedian(dvel))

# cast time series arrays
lsd_vels = np.array(lsd_vels, dtype=float)

lsd_flux = np.array(lsd_flux, dtype=float)
lsd_flux_err = np.array(lsd_flux_err, dtype=float)
lsd_fluxmodel = np.array(lsd_fluxmodel, dtype=float)

lsd_pol = np.array(lsd_pol, dtype=float)
lsd_pol_err = np.array(lsd_pol_err, dtype=float)

lsd_null = np.array(lsd_null, dtype=float)
lsd_null_err = np.array(lsd_null_err, dtype=float)

lsd_fluxmodel_corr, lsd_flux_corr, lsd_pol_corr,  lsd_null_corr = [], [], [], []
lsd_fluxerr_corr, lsd_polerr_corr, lsd_nullerr_corr = [], [], []

for i in range(len(inputdata)) :
    interp_fluxmodel_corr = interp1d(lsd_vels[i]-source_rvs[i], lsd_fluxmodel[i], kind='cubic')
    interp_flux_corr = interp1d(lsd_vels[i]-source_rvs[i], lsd_flux[i], kind='cubic')
    interp_pol_corr = interp1d(lsd_vels[i]-source_rvs[i], lsd_pol[i], kind='cubic')
    interp_null_corr = interp1d(lsd_vels[i]-source_rvs[i], lsd_null[i], kind='cubic')

    interp_fluxerr_corr = interp1d(lsd_vels[i]-source_rvs[i], lsd_flux_err[i], kind='cubic')
    interp_polerr_corr = interp1d(lsd_vels[i]-source_rvs[i], lsd_pol_err[i], kind='cubic')
    interp_nullerr_corr = interp1d(lsd_vels[i]-source_rvs[i], lsd_null_err[i], kind='cubic')

    lsd_fluxmodel_corr.append(interp_fluxmodel_corr(out_vels))
    lsd_flux_corr.append(interp_flux_corr(out_vels))
    lsd_pol_corr.append(interp_pol_corr(out_vels))
    lsd_null_corr.append(interp_null_corr(out_vels))

    lsd_fluxerr_corr.append(interp_fluxerr_corr(out_vels))
    lsd_polerr_corr.append(interp_polerr_corr(out_vels))
    lsd_nullerr_corr.append(interp_nullerr_corr(out_vels))


lsd_fluxmodel_corr = np.array(lsd_fluxmodel_corr, dtype=float)
lsd_flux_corr = np.array(lsd_flux_corr, dtype=float)
lsd_fluxerr_corr = np.array(lsd_fluxerr_corr, dtype=float)

lsd_pol_corr = np.array(lsd_pol_corr, dtype=float)
lsd_polerr_corr = np.array(lsd_polerr_corr, dtype=float)

lsd_null_corr = np.array(lsd_null_corr, dtype=float)
lsd_nullerr_corr = np.array(lsd_nullerr_corr, dtype=float)


# Polarimetry LSD Stokes V profiles -- RV corrected using the RV obtained from voigt model to the zeeman split:
#reduced_lsd_pol_corr = spu.subtract_median(lsd_pol_corr, vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=False, subtract=True)
reduced_lsd_pol_corr = spu.subtract_median(lsd_pol_corr, vels=out_vels, fit=True, verbose=False, median=True, subtract=True)
reduced_lsd_pol_corr = spu.subtract_median(reduced_lsd_pol_corr['ccf'], vels=out_vels, fit=True, verbose=False, median=True, subtract=True)
reduced_lsd_pol_corr = spu.subtract_median(reduced_lsd_pol_corr['ccf'], vels=out_vels, fit=True, verbose=False, median=True, subtract=True)


# set 2D plot parameters
if options.plot :
    x_lab = r"$Velocity$ [km/s]"     #Wavelength axis
    #y_lab = r"Time [BJD]"         #Time axis
    y_lab = r"Exposure number"         #Time axis
    z_lab_pol = r"Degree of polarization (Stokes V)"     #Intensity (exposures)
    z_lab_null = r"Null polarization"     #Intensity (exposures)
    z_lab_flux = r"Intensity (Stokes I)"     #Intensity (exposures)
    coolwarm_color_map = plt.cm.get_cmap('coolwarm')
    color_map = plt.cm.get_cmap('seismic')
    reversed_color_map = color_map.reversed()
    LAB_pol  = [x_lab,y_lab,z_lab_pol]
    LAB_null  = [x_lab,y_lab,z_lab_null]
    LAB_flux  = [x_lab,y_lab,z_lab_flux]

    spu.plot_2d(reduced_lsd_pol_corr['vels'], bjd, reduced_lsd_pol_corr['ccf'], LAB=LAB_pol, use_index_in_y=True, title="LSD Stokes V profiles", cmap=coolwarm_color_map)
    spu.plot_2d(reduced_lsd_pol_corr['vels'], bjd, reduced_lsd_pol_corr['residuals'], LAB=LAB_pol, use_index_in_y=True, title="LSD (Stokes V - Median) profiles", cmap=coolwarm_color_map)



for i in range(len(inputdata)) :
    output = inputdata[i].replace("_lsd.fits","_mslsd.fits")
    
    hdu = fits.open(inputdata[i])
    header = hdu[0].header
    header1 = hdu[1].header

    primary_hdu = fits.PrimaryHDU(header=header)

    hdu_vels = fits.ImageHDU(data=out_vels+source_rvs[i], name="Velocity", header=header1)
    hdu_pol = fits.ImageHDU(data=reduced_lsd_pol_corr['residuals'][i], name="StokesVQU")
    hdu_pol_err = fits.ImageHDU(data=lsd_polerr_corr[i], name="StokesVQU_Err")
    hdu_flux = fits.ImageHDU(data=lsd_flux_corr[i], name="StokesI")
    hdu_flux_err = fits.ImageHDU(data=lsd_fluxerr_corr[i], name="StokesI_Err")
    
    hdu_fluxmodel = fits.ImageHDU(data=lsd_fluxmodel_corr[i], name="StokesIModel")
    
    hdu_null = fits.ImageHDU(data=lsd_null_corr[i], name="Null")
    hdu_null_err = fits.ImageHDU(data=lsd_nullerr_corr[i], name="Null_Err")

    mef_hdu = fits.HDUList([primary_hdu, hdu_vels, hdu_pol, hdu_pol_err, hdu_flux, hdu_flux_err, hdu_fluxmodel, hdu_null, hdu_null_err])

    mef_hdu.writeto(output, overwrite=True)

    #if options.plot :
    #    plt.plot(out_vels,reduced_lsd_pol_corr['residuals'][i],alpha=0.2)
    
    hdu.close()

#if options.plot :
#    plt.plot(out_vels,reduced_lsd_pol_corr["ccf_med"],'-', lw=2)
#    plt.show()
