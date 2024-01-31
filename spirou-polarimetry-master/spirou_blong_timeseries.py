# -*- coding: iso-8859-1 -*-
"""
    Created on June 6 2020
    
    Description: Calculate longitudinal magnetic field (B-long) time series analysis for an input list of SPIRou LSD files
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python ~/spirou-tools/spirou-polarimetry/spirou_blong_timeseries.py --input=*_lsd.fits -pv -m
    python /Volumes/Samsung_T5/spirou-tools/spirou-polarimetry/spirou_blong_timeseries.py --input=2*_lsd.fits -pv -m
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
import matplotlib
import astropy.io.fits as fits

from scipy.interpolate import interp1d
from scipy import ndimage, misc

import spirouPolarUtils as spu

import spirouLSD


def measure_lsd_noise(inputdata, rv, fwhm, nsigclip=3, nfwhm=2.5, plot=False) :
    
    median_flux, median_pol, median_null = [], [], []
    sig_flux, sig_pol, sig_null = [], [], []
    bjd = []
    
    for i in range(len(inputdata)) :
        hdu = fits.open(inputdata[i])
        hdr = hdu[0].header + hdu[1].header
        if i == 0 :
            vels = hdu['VELOCITY'].data
            mask = vels < rv - nfwhm*fwhm
            mask ^= vels > rv + nfwhm*fwhm

        median_flux.append(np.nanmedian(hdu['STOKESI'].data[mask]))
        median_pol.append(np.nanmedian(hdu['STOKESVQU'].data[mask]))
        median_null.append(np.nanmedian(hdu['NULL'].data[mask]))
            
        sig_flux.append(np.nanstd(hdu['STOKESI'].data[mask]))
        sig_pol.append(np.nanstd(hdu['STOKESVQU'].data[mask]))
        sig_null.append(np.nanstd(hdu['NULL'].data[mask]))
        
        try :
            bjd.append(hdr["BJD"])
        except :
            bjd.append(hdr["BJDCEN"])

    loc = {}

    loc["median_flux"] = np.array(median_flux)
    loc["sig_flux"] = np.array(sig_flux)
    loc["median_pol"] = np.array(median_pol)
    loc["sig_pol"] = np.array(sig_pol)
    loc["median_null"] = np.array(median_null)
    loc["sig_null"] = np.array(sig_null)
    loc["BJD"] = np.array(bjd)

    median_sigpol = np.nanmedian(loc["sig_pol"])
    mad_sigpol = np.nanmedian(np.abs(loc["sig_pol"] - median_sigpol))  / 0.67449
    good = loc["sig_pol"] > median_sigpol - nsigclip*mad_sigpol
    good &= loc["sig_pol"] < median_sigpol + nsigclip*mad_sigpol
    
    loc["good"] = good

    if plot :
        minbjd, maxbjd = np.min(loc["BJD"]), np.max(loc["BJD"])
    
        plt.xlim(minbjd, maxbjd)
            
        plt.plot(median_sigpol)
        
        plt.plot(loc["BJD"][good], loc["sig_pol"][good], 'o', color='darkgreen')
        plt.plot(loc["BJD"][~good], loc["sig_pol"][~good], 'o', color='red', alpha=0.6)

        plt.hlines((median_sigpol),minbjd, maxbjd,ls="-",color="darkblue")
        plt.hlines((median_sigpol-nsigclip*mad_sigpol,median_sigpol+nsigclip*mad_sigpol),minbjd, maxbjd,ls="--",color="darkblue")
        plt.xlabel("BJD")
        plt.ylabel("polarimetric noise")
        
        plt.show()

    return loc



def measure_rvs_and_fwhm(inputdata, nsigclip=5, set_all_rvs_to_systemic=True, sysrv_type=1, vel_min=-1e50, vel_max=+1e50, plot=False, verbose=False) :
    
    lsd_flux = []
    rv, fwhm = np.array([]), np.array([])
    for i in range(len(inputdata)) :
        hdu = fits.open(inputdata[i])
        hdr = hdu[0].header + hdu[1].header
        if i == 0 :
            vels = hdu['VELOCITY'].data
        lsd_flux.append(hdu['STOKESI'].data)
        try :
            stokesI_fit = spu.fit_lsd_flux_profile(hdu['VELOCITY'].data, hdu['STOKESI'].data, hdu['STOKESI_ERR'].data, guess=None, func_type="gaussian", plot=False)
            rv = np.append(rv, stokesI_fit["VSHIFT"])
            fwhm = np.append(fwhm, 2.355 * stokesI_fit["SIG"])
        except :
            rv = np.append(rv, np.nan)
            fwhm = np.append(fwhm, np.nan)
            continue

    if verbose :
        print("FWHM={:.2f}+-{:.2f} km/s".format(np.nanmean(fmhm),np.nanstd(fmhm)))
    #plt.plot(fmhm)
    #plt.show()

    systemic_rv1 = np.nanmedian(rv)
    
    lsd_flux = np.array(lsd_flux, dtype=float)

    lsd_template = spu.subtract_median(lsd_flux, vels=vels, fit=True, verbose=False, median=True, subtract=True)

    min = np.argmin(lsd_template['ccf_med'])

    rv_min = vels[min]

    fitrange = vels - rv_min > vel_min
    fitrange &= vels - rv_min < vel_max

    try :
        median_stokesI_fit = spu.fit_lsd_flux_profile(vels[fitrange], lsd_template['ccf_med'][fitrange], lsd_template['ccf_sig'][fitrange], guess=None, func_type="gaussian", plot=plot)
        systemic_rv2 = median_stokesI_fit["VSHIFT"]
    except :
        systemic_rv2 = systemic_rv1

    sigma = np.nanmean(lsd_template["ccf_sig"])

    for i in range(len(inputdata)) :
        
        loc_sigma = np.nanstd(lsd_template["residuals"][i])
        
        if verbose :
            print("Exposure {0}/{1} SysRV={2:.3f} km/s RV={3:.3f} km/s rms={4:.1f} x sigma".format(i, len(inputdata), systemic_rv1, rv[i], loc_sigma/sigma))
        
        if loc_sigma/sigma < nsigclip :
            try :
                stokesI_fit = spu.fit_lsd_flux_profile(vels[fitrange], lsd_template['ccf'][i][fitrange], lsd_template['ccf_sig'][fitrange], guess=None, func_type="gaussian", plot=False)
                rv[i] = stokesI_fit["VSHIFT"]
            except :
                rv[i] = systemic_rv2
        else :
            rv[i] = np.nan
        #print(inputdata[i], i, rv[i])

    systemic_rv1 = np.nanmedian(rv)

    if set_all_rvs_to_systemic and sysrv_type == 1 :
        rv = np.full_like(rv,systemic_rv1)

    elif set_all_rvs_to_systemic and sysrv_type == 2 :
        
        fitrange = vels - systemic_rv1 > vel_min
        fitrange &= vels - systemic_rv1 < vel_max
        try :
            median_stokesI_fit = spu.fit_lsd_flux_profile(vels[fitrange], lsd_template['ccf_med'][fitrange], lsd_template['ccf_sig'][fitrange], guess=None, func_type="gaussian", plot=plot)
            systemic_rv2 = median_stokesI_fit["VSHIFT"]
        except :
            systemic_rv2 = systemic_rv1

        rv = np.full_like(rv,systemic_rv2)

    return rv, fwhm


def load_lsd_time_series(inputdata, constant_rv=False, nsigclip=5, fit_profile=False, vel_min=-1e50, vel_max=+1e50, auto_vel_range=True, auto_vel_nfwhm=3.5, verbose=False, plot=False) :
    
    loc = {}

    lsd_rv, lsd_fwhm = measure_rvs_and_fwhm(inputdata, nsigclip=nsigclip, set_all_rvs_to_systemic=constant_rv, vel_min=-20, vel_max=20)

    if auto_vel_range :
        fwhm, efwhm = np.nanmedian(lsd_fwhm), np.nanstd(lsd_fwhm)
        fullrange = np.nanmedian(lsd_fwhm) * auto_vel_nfwhm
        vel_min = - fullrange/2
        vel_max = + fullrange/2
        if verbose:
            print("Automatic velocity range results: FWHM={:.2f}+-{:.2f} km/s n={}x vel_min={:.2f} km/s  vel_max={:.2f} km/s".format(fwhm, efwhm, auto_vel_nfwhm, vel_min, vel_max))

        lsd_rv, lsd_fwhm = measure_rvs_and_fwhm(inputdata, nsigclip=nsigclip, set_all_rvs_to_systemic=constant_rv, vel_min=vel_min, vel_max=vel_max)

    maxrv, minrv = np.nanmax(lsd_rv), np.nanmin(lsd_rv)

    lsd_noise = measure_lsd_noise(inputdata, np.nanmedian(lsd_rv), np.nanmedian(lsd_fwhm), plot=plot)
    
    bjd = []
    airmass, snr = [], []
    waveavg, landeavg = [], []
    
    lsd_pol, lsd_null, lsd_flux  = [], [], []
    lsd_pol_err, lsd_flux_err  = [], []
    lsd_pol_corr, lsd_flux_corr  = [], []

    bfield = np.full(len(inputdata), np.nan)
    bfield_err = np.full(len(inputdata), np.nan)

    lsd_pol_gaussmodel = np.full(len(inputdata), None)
    lsd_pol_voigtmodel = np.full(len(inputdata), None)
    pol_rv = np.array(lsd_rv)
    zeeman_split = np.full(len(inputdata), np.nan)
    pol_line_depth = np.full(len(inputdata), np.nan)
    pol_fwhm = np.full(len(inputdata), np.nan)
 
    # Get velocity range from base exposure
    basehdu = fits.open(inputdata[0])
    vels_sup_lim = np.nanmax(basehdu['VELOCITY'].data)
    vels_inf_lim = np.nanmin(basehdu['VELOCITY'].data)
            
    if (vel_min + minrv) < vels_inf_lim :
        print("WARNING: requested RVs outside range, reseting vel_min to {:.1f} km/s".format(vels_inf_lim - minrv))
        vel_min = vels_inf_lim
            
    if (vel_max + maxrv) > vels_sup_lim :
        print("WARNING: requested RVs outside range, reseting vel_max to {:.1f} km/s".format(vels_sup_lim - maxrv))
        vel_max = vels_sup_lim
    
    mask = basehdu['VELOCITY'].data > vel_min
    mask &= basehdu['VELOCITY'].data < vel_max
    vels = basehdu['VELOCITY'].data[mask]

    for i in range(len(inputdata)) :
        
        if np.isnan(lsd_rv[i]) :
            if verbose:
                print("Rejecting LSD profile in file {0}/{1}: {2}".format(i, len(inputdata), inputdata[i]))
            continue
        
        if verbose:
            print("Loading LSD profile {0}/{1}: {2} ".format(i, len(inputdata), inputdata[i]))
        
        hdu = fits.open(inputdata[i])
        hdr = hdu[0].header + hdu[1].header

        if "MEANBJD" in hdr.keys() :
            bjd.append(float(hdr["MEANBJD"]))
        elif "BJD" in hdr.keys() :
            bjd.append(float(hdr["BJD"]))
        else :
            print("Could not read BJD from header, exit ...")
            exit()

        if "SNR33" in hdr.keys() :
            snr.append(float(hdr["SNR33"]))
        else :
            snr.append(1.0)

        airmass.append(float(hdr["AIRMASS"]))

        lsd_pol.append(hdu['STOKESVQU'].data[mask])
        lsd_null.append(hdu['NULL'].data[mask])
        lsd_flux.append(hdu['STOKESI'].data[mask])

        lsd_pol_err.append(hdu['STOKESVQU_ERR'].data[mask])
        lsd_flux_err.append(hdu['STOKESI_ERR'].data[mask])

        if fit_profile :
            try :
                # fit gaussian to the measured Stokes VQU LSD profile
                zeeman_gauss = spu.fit_zeeman_split(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], pol_err=hdu['STOKESVQU_ERR'].data[mask], func_type="gaussian", plot=False)
                
                lsd_pol_gaussmodel[i] = zeeman_gauss["MODEL"]
                
                try :
                    amplitude = zeeman_gauss["AMP"]
                    cont = zeeman_gauss["CONT"]
                    vel1 = zeeman_gauss["V1"]
                    vel2 = zeeman_gauss["V2"]
                    sigma = zeeman_gauss["SIG"]
                    guess = [amplitude, vel1, vel2, sigma, sigma, cont]

                    zeeman_voigt = spu.fit_zeeman_split(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], pol_err=hdu['STOKESVQU_ERR'].data[mask], guess=guess, func_type="voigt", plot=False)

                    lsd_pol_voigtmodel[i] = zeeman_voigt["MODEL"]
                    pol_rv[i] = zeeman_voigt["VSHIFT"]
                    zeeman_split[i] = zeeman_voigt["DELTAV"]
                    pol_line_depth[i] = zeeman_voigt["AMP"]
                    pol_fwhm[i] = zeeman_voigt["SIG"]
                except :
                    try :
                        zeeman_voigt = spu.fit_zeeman_split(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], pol_err=hdu['STOKESVQU_ERR'].data[mask], func_type="voigt", plot=False)

                        lsd_pol_voigtmodel[i] = zeeman_voigt["MODEL"]
                        pol_rv[i] = zeeman_voigt["VSHIFT"]
                        zeeman_split[i] = zeeman_voigt["DELTAV"]
                        pol_line_depth[i] = zeeman_voigt["AMP"]
                        pol_fwhm[i] = zeeman_voigt["SIG"]
                    except :
                        pol_rv[i] = zeeman_gauss["VSHIFT"]
                        pol_line_depth[i] = zeeman_gauss["AMP"]
                        pol_fwhm[i] = zeeman_gauss["SIG"]
                        
                        print("WARNING: could not fit Voigt function")
            except :
                print("WARNING: could not fit Gauss function")


        vels_corr = hdu['VELOCITY'].data - lsd_rv[i]

        pol_fit = interp1d(vels_corr, hdu['STOKESVQU'].data, kind='cubic')
        lsd_pol_corr.append(pol_fit(hdu['VELOCITY'].data[mask]))
        flux_fit = interp1d(vels_corr, hdu['STOKESI'].data, kind='cubic')
        lsd_flux_corr.append(flux_fit(hdu['VELOCITY'].data[mask]))

        b, berr = spu.longitudinal_b_field(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], hdu['STOKESI'].data[mask], hdr['WAVEAVG'], hdr['LANDEAVG'], pol_err=hdu['STOKESVQU_ERR'].data[mask], flux_err=hdu['STOKESI_ERR'].data[mask])

        landeavg.append(hdr['LANDEAVG'])
        waveavg.append(hdr['WAVEAVG'])

        bfield[i] = b
        bfield_err[i] = berr

        hdu.close()


    bjd = np.array(bjd)
    airmass, snr = np.array(airmass), np.array(snr)
    landeavg, waveavg = np.array(landeavg), np.array(waveavg)

    bfield, bfield_err = np.array(bfield), np.array(bfield_err)
    pol_rv, zeeman_split = np.array(pol_rv), np.array(zeeman_split)
    pol_line_depth, pol_fwhm = np.array(pol_line_depth), np.array(pol_fwhm)

    lsd_pol = np.array(lsd_pol, dtype=float)
    lsd_pol_err = np.array(lsd_pol_err, dtype=float)
    lsd_pol_corr = np.array(lsd_pol_corr, dtype=float)
    lsd_flux_corr = np.array(lsd_flux_corr, dtype=float)

    lsd_flux = np.array(lsd_flux, dtype=float)
    lsd_flux_err = np.array(lsd_flux_err, dtype=float)
    lsd_null = np.array(lsd_null, dtype=float)

    loc["SOURCE_RV"] = np.nanmedian(lsd_rv)
    loc["VELS"] = vels
    
    loc["BJD"] = bjd
    loc["AIRMASS"] = airmass
    loc["SNR"] = snr
    loc["WAVEAVG"] = waveavg
    loc["LANDEAVG"] = landeavg
    
    loc["LSD_FWHM"] = lsd_fwhm

    loc["LSD_RV"] = lsd_rv
    loc["POL_RV"] = pol_rv
    loc["ZEEMAN_SPLIT"] = zeeman_split
    loc["POL_LINE_DEPTH"] = pol_line_depth
    loc["POL_FWHM"] = pol_fwhm

    loc["BLONG"], loc["BLONG_ERR"] = bfield, bfield_err
    
    loc["LSD_POL"] = lsd_pol
    loc["LSD_FLUX"] = lsd_flux
    loc["LSD_FLUX_CORR"] = lsd_flux_corr
    loc["LSD_POL_ERR"] = lsd_pol_err
    loc["LSD_FLUX_ERR"] = lsd_flux_err
    loc["LSD_NULL"] = lsd_null
    loc["LSD_POL_CORR"] = lsd_pol_corr
    loc["LSD_POL_GAUSSMODEL"] = lsd_pol_gaussmodel
    loc["LSD_POL_VOIGTMODEL"] = lsd_pol_voigtmodel

    loc["LSD_FLUX_MEDIAN"] = lsd_noise["median_flux"]
    loc["LSD_FLUX_RMS"] = lsd_noise["sig_flux"]
    loc["LSD_POL_MEDIAN"] = lsd_noise["median_pol"]
    loc["LSD_POL_RMS"] = lsd_noise["sig_pol"]
    loc["LSD_NULL_MEDIAN"] = lsd_noise["median_null"]
    loc["LSD_NULL_RMS"] = lsd_noise["sig_null"]

    return loc


def calculate_blong_timeseries(lsddata, norm_errs=True, norm_errs_from_time_series=False, use_mc=False, use_corr_data=True, plot=False, debug=False) :

    bjd = lsddata["BJD"]
    vels = lsddata["VELS"]
    waveavg = lsddata["WAVEAVG"]
    landeavg = lsddata["LANDEAVG"]

    blong, blong_err = [], []
    
    ## First calculate possible residual continuum from the median profile
    #  and remove it from the data.
    pol_cont, pol_cont_err = continuum_lsd_I(lsddata["VELS"], lsddata["LSD_POL_MED"], lsddata["LSD_POL_MED_ERR"],fit_continuum=False, npcont=7, plot=False)
    flux_cont, flux_cont_err = continuum_lsd_I(lsddata["VELS"], lsddata["LSD_FLUX_MED"], lsddata["LSD_FLUX_MED_ERR"],fit_continuum=False, npcont=7, plot=False)
    
    if use_corr_data :
        lsd_pol = lsddata["LSD_POL_CORR"] - pol_cont
        lsd_flux = lsddata["LSD_FLUX_CORR"] / flux_cont
    else :
        lsd_pol = lsddata["LSD_POL"] - pol_cont
        lsd_flux = lsddata["LSD_FLUX"] / flux_cont

    if debug :
        for i in range(len(bjd)) :
            plt.plot(lsddata["VELS"], lsd_flux[i], '.', alpha=0.3)
        plt.plot(lsddata["VELS"], lsddata["LSD_FLUX_MED"]/flux_cont, '-', lw=2)
        plt.plot(lsddata["VELS"], np.full_like(lsddata["VELS"], 1.), '-', lw=2)
        plt.show()

        for i in range(len(bjd)) :
            plt.plot(lsddata["VELS"], lsd_pol[i], '.', alpha=0.3)
        plt.plot(lsddata["VELS"], lsddata["LSD_POL_MED"]-pol_cont, '-', lw=2)
        plt.plot(lsddata["VELS"], np.full_like(lsddata["VELS"], 0.), '-', lw=2)
        plt.show()
    ##-------------------
    
    for i in range(len(bjd)) :
        
        if norm_errs :
            median_polerr = np.nanmedian(lsddata["LSD_POL_ERR"][i])
            median_fluxerr = np.nanmedian(lsddata["LSD_FLUX_ERR"][i])

            if norm_errs_from_time_series :
                # adopt errors from the dispersion in the time series of all LSD data
                pol_err = (lsddata["LSD_POL_ERR"][i] / median_polerr) * lsddata["LSD_POL_MED_ERR"]
                flux_err = (lsddata["LSD_FLUX_ERR"][i] / median_fluxerr) * lsddata["LSD_FLUX_MED_ERR"] / flux_cont
            else :
                # adopt errors from the dispersion in the time series of all LSD data
                pol_err = (lsddata["LSD_POL_ERR"][i] / median_polerr) * lsddata["LSD_POL_RMS"][i]
                flux_err = (lsddata["LSD_FLUX_ERR"][i] / median_fluxerr) * lsddata["LSD_FLUX_RMS"][i] / flux_cont
        else :
            # adopt errors from LSD analysis
            pol_err = lsddata["LSD_POL_ERR"][i]
            flux_err = lsddata["LSD_FLUX_ERR"][i] / flux_cont
                
        if use_mc :
            # Below it calculates B-long using monte carlo, which gives a posterior distribution that's conditioned on the input data and uncertainties.
            b, berr = spu.longitudinal_b_field_montecarlo(vels, lsd_pol[i], pol_err, lsd_flux[i], flux_err, waveavg[i], landeavg[i], nsamples = 10000, plot=False, verbose=True)
        else :
            # Below it calculates B-long using the errors measured from the dispersion in the time-series
            b, berr = spu.longitudinal_b_field(vels, lsd_pol[i], lsd_flux[i], waveavg[i], landeavg[i], pol_err=pol_err, flux_err=flux_err)
        
        blong.append(b)
        blong_err.append(berr)
    
    blong = np.array(blong)
    blong_err = np.array(blong_err)

    lsddata["BLONG"], lsddata["BLONG_ERR"] = blong, blong_err

    bmed, bmederr = spu.longitudinal_b_field(vels, lsddata["LSD_POL_MED"] - pol_cont, lsddata["LSD_FLUX_MED"]/flux_cont, np.mean(waveavg), np.mean(landeavg), pol_err=lsddata["LSD_POL_MED_ERR"], flux_err=lsddata["LSD_FLUX_MED_ERR"]/flux_cont)

    if plot :
        font = {'size': 16}
        matplotlib.rc('font', **font)

        plt.errorbar(lsddata["BJD"], lsddata["BLONG"], yerr=lsddata["BLONG_ERR"], fmt='.', color="olive", label=r"B$_l$")
        
        plt.axhline(y=bmed-bmederr, ls='--', lw=1, color="orange")
        plt.axhline(y=bmed, ls='-', lw=2, color="blue", label=r"Mean B$_l$={0:.1f}+-{1:.1f} G".format(bmed, bmederr))
        plt.axhline(y=bmed+bmederr, ls='--', lw=1, color="orange")
        
        plt.plot(lsddata["BJD"], lsddata["BLONG"], '-', lw=0.7, color="olive")
        plt.ylabel("Longitudinal magnetic field [G]")
        plt.xlabel("BJD")
        plt.legend()
        plt.show()

    return lsddata


def save_blong_time_series(output, bjd, blong, blongerr, time_in_rjd=False) :
    
    sorted = np.argsort(bjd)
    
    outfile = open(output,"w+")
    
    for i in range(len(bjd[sorted])) :
        
        if time_in_rjd :
            time = bjd[sorted][i] - 2400000.
        else :
            time = bjd[sorted][i]
        
        outfile.write("{0:.10f} {1:.5f} {2:.5f}\n".format(time, blong[sorted][i], blongerr[sorted][i]))

    outfile.close()


def chi_square(lsddata, verbose=False) :

    meanblong = np.nanmean(lsddata["BLONG"])
    mblong = np.nanmedian(lsddata["BLONG"])
    blong_sig = np.nanstd(lsddata["BLONG"])
    blong_mad = np.nanmedian(np.abs(lsddata["BLONG"] - mblong))  / 0.67449
    dof = len(lsddata["BLONG"][np.isfinite(lsddata["BLONG"])])
    chisqr = np.nansum( ( (lsddata["BLONG"] - mblong)/lsddata["BLONG_ERR"] )**2)
    redchisqr = chisqr / dof
    if verbose :
        print("Chi-square={:.3f}; DOF={} ; Reduced chi-square={:.3f}; Constant model: Bl_median={:.2f}+/-{:.2f} G Bl_mean={:.2f}+/-{:.2f} G".format(chisqr, dof, redchisqr, mblong, blong_mad, meanblong, blong_sig))

    return chisqr, dof, redchisqr


def reduce_lsddata(lsddata, niter=3, apply_median_filter=True, median_filter_size=3, use_residuals=False, plot=False) :
    
    bjd = lsddata["BJD"]
    vels = lsddata["VELS"]

    if plot :
        x_lab = r"$Velocity$ [km/s]"     #Wavelength axis
        y_lab = r"Time [BJD]"         #Time axis
        z_lab_pol = r"Degree of polarization (Stokes V)"     #Intensity (exposures)
        z_lab_null = r"Null polarization (Stokes V)"     #Intensity (exposures)
        z_lab_flux = r"Intensity (Stokes I)"     #Intensity (exposures)
        LAB_pol  = [x_lab,y_lab,z_lab_pol]
        LAB_null  = [x_lab,y_lab,z_lab_null]
        LAB_flux  = [x_lab,y_lab,z_lab_flux]

    lsd_pol_corr = spu.subtract_median(lsddata["LSD_POL_CORR"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_flux_corr = spu.subtract_median(lsddata["LSD_FLUX_CORR"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_pol = spu.subtract_median(lsddata["LSD_POL"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_flux = spu.subtract_median(lsddata["LSD_FLUX"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_null = spu.subtract_median(lsddata["LSD_NULL"] - np.median(lsddata["LSD_NULL"]), vels=vels, fit=True, verbose=False, median=True, subtract=True)

    # Polarimetry LSD Stokes V profiles:
    for iter in range(niter) :
        lsd_pol = spu.subtract_median(lsd_pol['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_flux = spu.subtract_median(lsd_flux['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_pol_corr = spu.subtract_median(lsd_pol_corr['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_flux_corr = spu.subtract_median(lsd_flux_corr['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_null = spu.subtract_median(lsd_null['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)

    if use_residuals :
        lsd_pol = spu.subtract_median(lsd_pol['residuals'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_pol_corr = spu.subtract_median(lsd_pol_corr['residuals'], vels=vels, fit=True, verbose=False, median=True, subtract=True)

    if plot :
        spu.plot_2d(lsd_pol_corr['vels'], bjd, lsd_pol_corr['ccf'], LAB=LAB_pol, title="LSD Stokes V profiles", cmap="seismic")
        spu.plot_2d(lsd_flux_corr['vels'], bjd, lsd_flux_corr['ccf'], LAB=LAB_flux, title="LSD Stokes I profiles", cmap="seismic")

    if apply_median_filter :
        lsd_pol_medfilt = ndimage.median_filter(lsd_pol['ccf'], size=median_filter_size)
        lsd_flux_medfilt = ndimage.median_filter(lsd_flux['ccf'], size=median_filter_size)
        lsd_pol_corr_medfilt = ndimage.median_filter(lsd_pol_corr['ccf'], size=median_filter_size)
        lsd_flux_corr_medfilt = ndimage.median_filter(lsd_flux_corr['ccf'], size=median_filter_size)
        lsd_null_medfilt = ndimage.median_filter(lsd_null['ccf'], size=median_filter_size)

        if plot :
            spu.plot_2d(lsd_pol_corr['vels'], bjd, lsd_pol_corr_medfilt, LAB=LAB_pol, title="Median-filtered LSD Stokes V profiles", cmap="seismic")
            spu.plot_2d(lsd_flux_corr['vels'], bjd, lsd_flux_corr_medfilt, LAB=LAB_flux, title="Median-filtered LSD Stokes I profiles", cmap="seismic")

        lsddata["LSD_POL"] = lsd_pol_medfilt
        lsddata["LSD_FLUX"] = lsd_flux_medfilt
        lsddata["LSD_POL_CORR"] = lsd_pol_corr_medfilt
        lsddata["LSD_FLUX_CORR"] = lsd_flux_corr_medfilt
        lsddata["LSD_NULL"] = lsd_null_medfilt

    else :
        lsddata["LSD_POL"] = lsd_pol['ccf']
        lsddata["LSD_FLUX"] = lsd_flux['ccf']
        lsddata["LSD_POL_CORR"] = lsd_pol_corr['ccf']
        lsddata["LSD_FLUX_CORR"] = lsd_flux_corr['ccf']
        lsddata["LSD_NULL"] = lsd_null['ccf']

    lsddata["LSD_POL_MED"] = lsd_pol_corr['ccf_med']
    lsddata["LSD_POL_MED_ERR"] = lsd_pol_corr['ccf_sig']
    lsddata["LSD_FLUX_MED"] = lsd_flux_corr['ccf_med']
    lsddata["LSD_FLUX_MED_ERR"] = lsd_flux_corr['ccf_sig']

    return lsddata


def continuum_lsd_I(vels, flux, fluxerr, fit_continuum=True, npcont=10, plot=False) :
    
    cont_sample = np.append(flux[:npcont],flux[-npcont:])
    cont_err_sample = np.append(fluxerr[:npcont],fluxerr[-npcont:])
    cont_vels = np.append(vels[:npcont],vels[-npcont:])
    
    if fit_continuum :
        c = np.polyfit(cont_vels, cont_sample, 1)
        p = np.poly1d(c)
        cont = p(vels)
    else :
        c = np.nanmedian(cont_sample)
        cont = np.full_like(flux, c)

    err = np.full_like(fluxerr,np.nanmedian(cont_err_sample))
    
    if plot :
        # plot flux profile to check continuum
        plt.errorbar(vels, flux, fluxerr, fmt='.')
        plt.plot(cont_vels, cont_sample, 'o')
        plt.plot(vels, cont, '--')
        plt.show()

    return cont, err


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input LSD data pattern",type='string',default="*_lsd.fits")
parser.add_option("-o", "--output", dest="output", help="Output B-long time series file",type='string',default="")
parser.add_option("-1", "--min_vel", dest="min_vel", help="Minimum velocity [km/s]",type='float',default=0.)
parser.add_option("-2", "--max_vel", dest="max_vel", help="Maximum velocity [km/s]",type='float',default=0.)
parser.add_option("-s", "--nsigclip", dest="nsigclip", help="Threshold in number of sigmas to keep LSD",type='float',default=5.)
parser.add_option("-n", "--nfwhm", dest="nfwhm", help="LSD velocity range in number of FWHMs to integrate B-long",type='float',default=7.)
parser.add_option("-e", action="store_true", dest="use_residuals", help="use residual Stokes VQU profiles", default=False)
parser.add_option("-m", action="store_true", dest="use_mc", help="Use Monte Carlo sampling to calculate final values of B-long", default=False)
parser.add_option("-c", action="store_true", dest="constant_rv", help="Set all profiles with a constant velocity", default=False)
parser.add_option("-f", action="store_true", dest="median_filter", help="Apply median filter to polar time series", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with spirou_blong_timeseries.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input LSD data pattern: ', options.input)
    print('Output Blong time series file: ', options.output)
    if options.min_vel :
        print('Minimum velocity = {0:.2f} km/s: '.format(options.min_vel))
    if options.min_vel :
        print('Maximum velocity = {0:.2f} km/s: '.format(options.min_vel))
    print('Threshold in number of sigmas to keep LSD: {0:.0f}'.format(options.nsigclip))
    print('LSD velocity range in number of FWHMs to integrate B-long: {0:.0f}'.format(options.nfwhm))

# make list of data files
if options.verbose:
    print("Creating list of lsd files...")
inputdata = sorted(glob.glob(options.input))

#---
if options.min_vel != 0 and options.max_vel != 0 :
    auto_vel_range = False
else :
    auto_vel_range = True
    if options.min_vel == 0:
        options.min_vel = -35.
    if options.max_vel == 0:
        options.max_vel = +35.

norm_errs = True

lsddata = load_lsd_time_series(inputdata, constant_rv=options.constant_rv, nsigclip=options.nsigclip, fit_profile=False, vel_min=options.min_vel, vel_max=options.max_vel, auto_vel_range=auto_vel_range, auto_vel_nfwhm=options.nfwhm, verbose=options.verbose, plot=options.plot)

lsddata = reduce_lsddata(lsddata, apply_median_filter=options.median_filter, median_filter_size=(5,2), use_residuals=options.use_residuals, plot=False)

lsddata = calculate_blong_timeseries(lsddata, norm_errs=norm_errs, norm_errs_from_time_series=False, use_mc=options.use_mc, use_corr_data=True, plot=options.plot)

chisqr, dof, redchisqr = chi_square(lsddata, verbose=options.verbose)

if options.output != "" :
    save_blong_time_series(options.output, lsddata["BJD"], lsddata["BLONG"], lsddata["BLONG_ERR"])
