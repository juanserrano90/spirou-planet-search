# -*- coding: iso-8859-1 -*-
"""
    Created on April 8 2021
    
    Description: This routine run spirou_lsd.py on a series of SPIRou polarimetric spectra.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-polarimetry/spirou_lsd_pipeline.py --input=*p.fits --lsdmask=/Users/eder/spirou-tools/spirou-polarimetry/lsd_masks/marcs_t5000g50_atom -v
    
    python ~/spirou-tools/spirou-polarimetry/spirou_lsd_pipeline.py --input=*p.fits -v
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input polarimetry p.fits data pattern",type='string',default="*p.fits")
parser.add_option("-m", "--lsdmask", dest="lsdmask", help="Input LSD mask",type='string',default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_lsd_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('Spectral p.fits data pattern: ', options.input)
    print('LSD mask: ', options.lsdmask)

spirou_pol_dir = os.path.dirname(__file__) + '/'

# make list of efits data files
if options.verbose:
    print("Creating list of p.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

lsdmask_str = ""
if options.lsdmask != "":
    lsdmask_str = " --lsdmask={0}".format(options.lsdmask)

verbose_flag = ""
if options.verbose :
    verbose_flag = " -v"

for i in range(len(inputdata)) :
    
    output_lsd =inputdata[i].replace(".fits","_lsd.fits")
    
    command = "python {0}spirou_lsd.py --input={1} --output={2}{3}{4}".format(spirou_pol_dir, inputdata[i], output_lsd, lsdmask_str, verbose_flag)

    print("Running: ",command)
    os.system(command)

