"""
Script to take file with C_L^\kappa values and compute resulting bias on
a determination of neutrino mass.

Example usage (from pyfisher root directory):
    python bin/mnu_baryon_bias.py S4 lensing output/test_ratio.dat -tau_prior 0.01

The file specified in the kappaRatioFile argument should contain lists
of ell values and C_L^kappa ratio values. The following command will
generate a file in the correct format:
    np.savetxt('test_ratio.dat',(ell,ratio))
where ell and ratio are 1d arrays of ell and ratio values, respectively.
"""

########################
# Imports
########################

# System
import sys, os
from configparser import ConfigParser
import pickle
import numpy as np
from scipy.interpolate import interp1d
import argparse

# orphics
from orphics.io import dict_from_section, list_from_config, cprint, Plotter
from orphics.cosmology import LensForecast
from orphics import cosmology

# pyfisher
from pyfisher.lensInterface import lensNoise
from pyfisher.clFisher import tryLoad, calcFisher, loadFishers, noiseFromConfig, rSigma
import pyfisher.clFisher as clf


########################
# Constants
########################

# Set CMB temperature
TCMB = 2.7255e6 # K


########################
# Initial inputs
########################

# Get the name of the experiment and lensing type from command line
parser = argparse.ArgumentParser(description='Compute neutrino mass bias from Fisher formalism')
parser.add_argument('expName', type=str,
                    help='The name of the experiment in .ini file')
parser.add_argument('lensName',type=str,default="",
                    help='The name of the CMB lensing section in .ini file',)
parser.add_argument('kappaRatioFile',type=str,default="",nargs='?',
                    help='The name of the file containing the convergence'
                         'power spectrum ratio to use for the parameter bias computation')
parser.add_argument("-t", "--tt",type=str,default=None,
                    help="Dimensionless beam-deconvolved TT noise curve file"
                         " to override experiment.")
parser.add_argument("-p", "--pp",type=str,default=None,
                    help="Dimensionless beam-deconvolved PP (EE/BB) noise "
                         "curve file to override experiment.")
parser.add_argument('-tau_prior','--tau_prior',type=float,default=None,
                    help='External tau prior')
args = parser.parse_args()
expName = args.expName
lensName = args.lensName
kappaRatioFile = args.kappaRatioFile
tau_prior = args.tau_prior

# Read config file
iniFile = "input/params_mnu_baryon_bias.ini"
Config = ConfigParser()
Config.optionxform=str
Config.read(iniFile)

# If specified on command line, load TT noise
try:
    elltt,ntt = np.loadtxt(args.tt,usecols=[0,1],unpack=True)
    noise_func_tt = interp1d(elltt,ntt,bounds_error=False,fill_value=np.inf)
except:
    noise_func_tt = None

# If specified on command line, load polarization noise
try:
    ellee,nee = np.loadtxt(args.pp,usecols=[0,1],unpack=True)
    noise_func_ee = interp1d(ellee,nee,bounds_error=False,fill_value=np.inf)
except:
    noise_func_ee = None
    
# Set file root name for Fisher derivatives
derivRoot = Config.get("fisher","derivRoot")

# Get list of parameters
paramList = Config.get("fisher","paramList").split(',')

    
########################
# Load and prepare spectra
########################

efficiencies = []
mnus = []
sns = []
rs = []
rdelens = []

# Get lensing noise curve.
# To override something from the Config file in order to make plots varying it,
# change from None to the value you want.
ls,Nls,ellbb,dlbb,efficiency,cc = lensNoise(Config,expName,lensName,
                                            beamOverride=None,noiseTOverride=None,
                                            lkneeTOverride=None,lkneePOverride=None,
                                            alphaTOverride=None,alphaPOverride=None,
                                            noiseFuncT=noise_func_tt,noiseFuncP=noise_func_ee)

# Print delensing efficiency
if False:
    cprint("Delensing efficiency: "+ str(efficiency) + " %",color="green",bold=True)

# Load fiducial spectra and and derivatives
fidCls = tryLoad(derivRoot+'_fCls.csv',',')
dCls = {}
for paramName in paramList:
    dCls[paramName] = tryLoad(derivRoot+'_dCls_'+paramName+'.csv',',')
    
# Load other Fisher matrices to add
otherFisher = loadFishers(Config.get('fisher','otherFishers').split(','))

# Get CMB noise power spectra and ell ranges
if (noise_func_tt is None) or (noise_func_ee is None):
    fnTT, fnEE = noiseFromConfig(Config,expName,TCMB=TCMB,
                                 beamsOverride=None,noisesOverride=None,
                                 lkneeTOverride=None,lkneePOverride=None,
                                 alphaTOverride=None,alphaPOverride=None)

# Get temperature and polarization ell ranges
tellmin,tellmax = list_from_config(Config,expName,'tellrange')
pellmin,pellmax = list_from_config(Config,expName,'pellrange')

# Pad noise with infinite values at high ell
if (noise_func_tt is not None):
    fnTT = cosmology.noise_pad_infinity(noise_func_tt,tellmin,tellmax)
if (noise_func_ee is not None):
    fnEE = cosmology.noise_pad_infinity(noise_func_ee,pellmin,pellmax)

# Pad CMB lensing noise with infinity outside L ranges
kellmin,kellmax = list_from_config(Config,'lensing','Lrange')
fnKK = cosmology.noise_pad_infinity(interp1d(ls,Nls,
                                             fill_value=np.inf,
                                             bounds_error=False),
                                    kellmin,kellmax)

# Determine ell range for Fisher matrix calculation
ellrange = np.arange(min(tellmin,pellmin,kellmin),max(tellmax,pellmax,kellmax)).astype(int)

# Load C_L^kappa ratio curve that we'll use for parameter bias computation
if kappaRatioFile == '':
    print('No input C_L^kappa ratio file specified - will only compute mnu errorbar')
    bary_ratio_ell = ellrange
    bary_ratio_cl_input = np.ones_like(ellrange)
else:
    print('Using lensing power spectrum ratio from %s' % kappaRatioFile)
    bary_ratio_ell,bary_ratio_cl_input = np.loadtxt(kappaRatioFile)
    
# Make interpolating function for C_L^kappa ratio, 
# and make array of \Delta C_L that is zero except for the change
# in C_L^kappa
bary_ratio_interp = interp1d(bary_ratio_ell,bary_ratio_cl_input,
                             bounds_error=False,fill_value=1.)
DeltaCls = np.zeros_like(fidCls)
DeltaCls[:,4] = (bary_ratio_interp(range(len(fidCls))) - 1.) * fidCls[:,4]

# Get fsky
fsky = Config.getfloat(expName,'fsky')


########################
# Compute S/N on lensing auto spectrum
########################

Clkk = fidCls[:,4]
frange = np.array(range(len(Clkk)))
snrange = np.arange(kellmin,kellmax)
LF = LensForecast()
LF.loadKK(frange,Clkk,ls,Nls)
kksn,errs = LF.sn(snrange,fsky,"kk")
cprint('Lensing auto power S/N: %g' % kksn,color="red",bold=True)


########################
# Compute Fisher matrix
########################

# Compute the Fisher matrix
cmbFisher = calcFisher(paramList,ellrange,fidCls,dCls,
                       lambda x: fnTT(x)*TCMB**2.,
                       lambda x: fnEE(x)*TCMB**2.,
                       fnKK,fsky,verbose=False)
totalFisher = otherFisher + cmbFisher
# totalFisher = cmbFisher

# Get prior sigmas and add to Fisher
priorList = Config.get("fisher","priorList").split(',')
for prior,param in zip(priorList,paramList):
    try:
        priorSigma = float(prior)
    except ValueError:
        continue
    ind = paramList.index(param)
    totalFisher[ind,ind] += 1./priorSigma**2.
    
# Get tau prior specified on command line
if tau_prior is not None:
    print('External tau prior: %g' % tau_prior)
    ind = paramList.index('tau')
    totalFisher[ind,ind] += 1./tau_prior**2.
else:
    print('NO external tau prior!')

# Get index of mnu and print marginalized constraint
indMnu = paramList.index('mnu')
mnu = np.sqrt(np.linalg.inv(totalFisher)[indMnu,indMnu])*1000.
cprint("1-sigma uncertainty on sum of neutrino masses: %g meV" % mnu,color="green",bold=True)

# Compute parameter bias fector and propagate to bias on mnu
print("Calculating bias vector...")
Bvec = clf.calcBvec(paramList,ellrange,DeltaCls,fidCls,dCls,
                    fnTT,fnEE,fnKK,fsky,verbose=False)
b = clf.calcBias(totalFisher,Bvec)[indMnu][0]

# Print mnu bias information
cprint('Neutrino mass biases:',color="green",bold=True)
print('\t%g meV' % (b*1000.))   
print('\t%g-sigma' % (b*1000./mnu))
print('\t%g %% of statistical error' % (b*100.*1000./mnu))
print('\t%g %% of minimal value' % (b*100.*1000./60.))
