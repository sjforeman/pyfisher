import camb
from camb import model, initialpower
import numpy as np
import sys, os
from configparser import ConfigParser
from orphics import mpi

from pyfisher.core import getHubbleCosmology

    
def getPowerCamb(params,spec='lensed_scalar',AccuracyBoost=False):
    '''
    Get Cls from CAMB
    '''
    
    pars = camb.CAMBparams()
    if AccuracyBoost:
        pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=2.0, lAccuracyBoost=2.0)
        # pars.set_accuracy(AccuracyBoost=3.0, lSampleBoost=3.0, lAccuracyBoost=3.0)
    pars.set_dark_energy(w=params['w'],wa=params['wa'],dark_energy_model='ppf')
    try:
        theta = params['theta100']/100.
        H0 = None
    except:
        H0 = params['H0']
        theta = None
    pars.set_cosmology(H0=H0, cosmomc_theta=theta, ombh2=params['ombh2'], omch2=params['omch2'], tau=params['tau'],mnu=params['mnu'],nnu=params['nnu'],omk=params['omk'],num_massive_neutrinos=int(params['num_massive_neutrinos']),TCMB=params['TCMB'])
    pars.InitPower.set_params(As=params['As'],ns=params['ns'],r=params['r'],pivot_scalar=params['s_pivot'], pivot_tensor=params['t_pivot'])
    #pars.set_for_lmax(lmax=int(params['lmax']), lens_potential_accuracy=1, max_eta_k=2*params['lmax'])
    #pars.set_for_lmax(lmax=int(params['lmax']), lens_potential_accuracy=2.0, max_eta_k=50000.0)
    pars.set_for_lmax(lmax=int(params['lmax']), lens_potential_accuracy=3.0, max_eta_k=50*params['lmax'])

    #WantTensors
    if params['r']:
        pars.WantTensors = True
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars)

    if spec=='tensor':
        CL = results.get_tensor_cls(int(params['lmax']))*1.e12*params['TCMB']**2.
    else:    
        CL=powers[spec]*1.e12*params['TCMB']**2.

    lensArr = results.get_lens_potential_cls(lmax=int(params['lmax']))
    clphi = lensArr[:,0]
    cltphi = lensArr[:,1]

    j=0
    CLS = []
    # numpify this!
    for row in CL:
        if j>params['lmax']: break
        raw = row*2.*np.pi/(j*(j+1.))
        rawkk = clphi[j]* (2.*np.pi/4.)
        #rawkt = cltphi[j]*2.*np.pi/4.
        # correct KT (l^2*ClphiT and dimensional - uK)
        rawkt = cltphi[j]* (2.*np.pi/2.) / np.sqrt(j*(j+1.)) * params['TCMB']*1.e6
        CLS.append(np.append(raw,(rawkk,rawkt)))
                   
        j+=1

    CLS = np.nan_to_num(np.array(CLS))

    return CLS


#def main(argv):

if True:

    # Read Config
    iniFile = os.environ['FISHER_DIR']+"/input/" + sys.argv[1] #"makeDefaults_szar.ini"
    #iniFile = os.environ['FISHER_DIR']+'/input/June2_makeDerivs_optimal.ini'
    Config = ConfigParser.SafeConfigParser()
    Config.optionxform = str
    Config.read(iniFile)
    spec = Config.get('general','spec')
    out_pre = Config.get('general','output_prefix')
    AccuracyBoost = Config.getboolean('general','AccuracyBoost')
    try:
        derivForm = Config.getint('general','derivForm')
    except:
        derivForm = 0
    paramList = []
    fparams = {}
    stepSizes = {}
    fidscript = ''
    for (key, val) in Config.items('camb'):
        if ',' in val:
            param, step = val.split(',')
            paramList.append(key)
            fparams[key] = float(param)
            stepSizes[key] = float(step)
        else:
            fparams[key] = float(val)
        fidscript += key+' = '+val+'\n'
    
    # Save fid + stepsize
    fidscript = '[camb]\n'+fidscript
    filename = os.environ['FISHER_DIR']+"/output/"+out_pre+'_'+spec+"_fid.csv"
    with open(filename,'w') as tempFile:
        tempFile.write(fidscript)


    comm = mpi.MPI.COMM_WORLD
    rank = comm.Get_rank()
    numcores = comm.Get_size()
    Njobs = len(paramList)
    num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
    if rank==0: print("At most ", max(num_each) , " tasks...")
    my_tasks = each_tasks[rank]
        
    print("Rank ", rank , " reporting for duty")
    # Save fiducials
    if rank==0:
        print("Calculating and saving fiducial cosmology...")
        # if not('H0' in fparams):
        #     fparams['H0'] = getHubbleCosmology(theta=fparams['theta100'],params=fparams)
        fidCls = getPowerCamb(fparams,spec,AccuracyBoost=AccuracyBoost)
        #fidCls = getPowerCamb(fparams,spec+"_scalar",AccuracyBoost=AccuracyBoost)
        np.savetxt(os.environ['FISHER_DIR']+"/output/"+out_pre+'_'+spec+"_fCls.csv",fidCls,delimiter=",")


    
    #for paramName in paramList:
    
    for task in my_tasks:
    
        # Calculate and save derivatives
        paramName = paramList[task]
        
        print("Calculating derivatives for ", paramName)
        if derivForm == 0:
            print('Using 2-point stencil')
            h = stepSizes[paramName]
            
            print("Calculating forward difference for ", paramName)
            pparams = fparams.copy()
            pparams[paramName] = fparams[paramName] + 0.5*h
            # if paramName=='theta100':
            #     pparams['H0'] = getHubbleCosmology(theta=pparams['theta100'],params=pparams)
            pCls = getPowerCamb(pparams,spec,AccuracyBoost=AccuracyBoost)
            #pCls = getPowerCamb(pparams,spec+"_scalar",AccuracyBoost=AccuracyBoost)
    
            print("Calculating backward difference for ", paramName)
            mparams = fparams.copy()
            mparams[paramName] = fparams[paramName] - 0.5*h
            # if paramName=='theta100':
            #     mparams['H0'] = getHubbleCosmology(theta=mparams['theta100'],params=mparams)
            mCls = getPowerCamb(mparams,spec,AccuracyBoost=AccuracyBoost)
            #mCls = getPowerCamb(mparams,spec+"_scalar",AccuracyBoost=AccuracyBoost)
            
            dCls = (pCls-mCls)/h
            
        elif derivForm == 1:
            print('Using 5-point stencil')
            h = 0.5*stepSizes[paramName]
            
            params1 = fparams.copy()
            params2 = fparams.copy()
            params3 = fparams.copy()
            params4 = fparams.copy()
            params1[paramName] = fparams[paramName] + 2.*h
            params2[paramName] = fparams[paramName] + h
            params3[paramName] = fparams[paramName] - h
            params4[paramName] = fparams[paramName] - 2.*h
            # if paramName=='theta100':
            #     params1['H0'] = getHubbleCosmology(theta=params1['theta100'],params=params1)
            #     params2['H0'] = getHubbleCosmology(theta=params2['theta100'],params=params2)
            #     params3['H0'] = getHubbleCosmology(theta=params3['theta100'],params=params3)
            #     params4['H0'] = getHubbleCosmology(theta=params4['theta100'],params=params4)
            Cls1 = getPowerCamb(params1,spec,AccuracyBoost=AccuracyBoost)
            Cls2 = getPowerCamb(params2,spec,AccuracyBoost=AccuracyBoost)
            Cls3 = getPowerCamb(params3,spec,AccuracyBoost=AccuracyBoost)
            Cls4 = getPowerCamb(params4,spec,AccuracyBoost=AccuracyBoost)
            
            dCls = (- Cls1 + 8.*Cls2 - 8.*Cls3 + Cls4)/(12.*h)
    
        np.savetxt(os.environ['FISHER_DIR']+"/output/"+out_pre+'_'+spec+"_dCls_"+paramName+".csv",dCls,delimiter=",")

        if rank==0: print("Rank 0 done with task ", task+1, " / " , len(my_tasks))

