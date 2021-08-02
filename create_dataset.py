import os
import numpy as np
import nibabel as nib
import DCE_matt as matt
import hyperparams
import pandas as pd
import pickle
import argparse
from tqdm import tqdm

rootdir = '/scratch/tottens/DCE_MRI/data/'

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='MIPA')
parser.add_argument('--ip', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()

# Compute parameters of AIF according to Cosine8AIF
AIF = np.loadtxt('AIF.txt')
hp.acquisition.timing = np.arange(160) * hp.simulations.time / 60
aif = matt.fit_aif(AIF, hp.acquisition.timing)
hp.aif.aif = aif

# Create dict of Hct from all patients
if args.dataset == 'MIPA':
    Hct_data_OK = np.array(pd.read_excel(rootdir+'HCT.xlsx', sheet_name='Sheet1'))
    Hct_data_CR1 = np.array(pd.read_excel(rootdir+'HCT.xlsx', sheet_name='Sheet2'))
    Hct_data_CR2 = np.array(pd.read_excel(rootdir+'HCT.xlsx', sheet_name='Sheet3'))
    Hct_data_CR = np.insert(Hct_data_CR2, np.arange(len(Hct_data_CR1)), Hct_data_CR1, axis=0)
    Hct_data = np.concatenate((Hct_data_CR, Hct_data_OK), axis=0)
    Hct_data[Hct_data == 0] = 0.4
    Hct_data = dict(Hct_data)

elif args.dataset == 'REPRO':
    Hct_data = {}
    Hct_data_REMP = np.array(pd.read_excel(rootdir+'REMP_patienten.xlsx')['HCTcalc'])
    for i in range(1, 11):
        for j in range(1, 3):
            patient_num = 'REMP{:02d}_{}'.format(i, j)
            Hct_data[patient_num] = Hct_data_REMP[i]

else:
    raise ValueError('not a valid dataset, choice between MIPA or REPRO')

file_map = {'MIPA': 'CR', 'REPRO': 'REMP'}

print(Hct_data)


# Interpolation to fill in missing points in signal
def interpolation(Signal_all):
    for i in tqdm(range(Signal_all.shape[0])):
        for j in range(Signal_all.shape[1]):
            if np.mean(Signal_all[i]) == 0:
                continue

            if Signal_all[i][j] == 0:
                prev_non_0 = j-1
                next_non_0 = j+1
                if prev_non_0 < 0:
                    prev_non_0 = np.nan

                else:
                    while Signal_all[i][prev_non_0] == 0:
                        prev_non_0 -= 1
                        if prev_non_0 < 0:
                            prev_non_0 = np.nan
                            break

                if next_non_0 > Signal_all.shape[1]-1:
                    next_non_0 = np.nan

                else:
                    while Signal_all[i][next_non_0] == 0:
                        next_non_0 += 1
                        if next_non_0 > Signal_all.shape[1]-1:
                            next_non_0 = np.nan
                            break

                if not np.isnan(prev_non_0) and not np.isnan(next_non_0):
                    Signal_all[i][j] = (Signal_all[i][prev_non_0] + Signal_all[i][next_non_0])/2

                elif np.isnan(prev_non_0):
                    Signal_all[i][j] = Signal_all[i][next_non_0]

                elif np.isnan(next_non_0):
                    Signal_all[i][j] = Signal_all[i][prev_non_0]

    return Signal_all


# Initialize arrays of patient data
C1s = np.array([])
T1s = np.array([])
hcts = np.array([])
masks = np.array([])

counter = 0
first_patient = True
slices_per_patient = {}

for patient_dir in sorted(os.listdir(rootdir)):
    if file_map[args.dataset] in patient_dir:
        print(patient_dir)
        file_path = rootdir+patient_dir+'/result.0.nii.gz'

        if os.path.exists(rootdir+patient_dir+'/dceT1.nii.gz'):
            file_path_T1 = rootdir+patient_dir+'/dceT1.nii.gz'
        elif os.path.exists(rootdir+patient_dir+'/T1dce.nii.gz'):
            file_path_T1 = rootdir+patient_dir+'/T1dce.nii.gz'
        else:
            print('T1 not found')
            continue

        T1_orig = np.moveaxis(nib.load(file_path_T1).get_fdata(), 2, 0)
        signal_orig = np.moveaxis(nib.load(file_path).get_fdata(), 2, 0)

        invalid_T1 = []
        for i in range(len(T1_orig)):
            if np.mean(T1_orig[i]) < 1:
                invalid_T1.append(i)

        T1 = np.delete(T1_orig, invalid_T1, axis=0)
        signal = np.delete(signal_orig, invalid_T1, axis=0)

        slices_per_patient[patient_dir] = np.arange(len(signal)) + counter
        counter += len(signal)

        original_size = signal.shape
        signal = np.reshape(signal, (-1, signal.shape[-1]))
        T1 = T1.flatten()

        T1[T1 < 300] = 1e9
        T1[T1 > 6000] = 6000
        T1 /= 1000

        S0 = np.mean(signal[:, :hp.acquisition.rep0], axis=1)
        signal[S0 < 10] = 1

        if args.ip:
            signal = interpolation(signal)

        S0 = np.mean(signal[:, :hp.acquisition.rep0], axis=1)
        R1map = 1/T1

        R1eff = matt.dce_to_r1eff(signal, S0, R1map, hp.acquisition.TR, hp.acquisition.FA2)
        R1map = np.expand_dims(R1map, axis=1)
        C1 = matt.r1eff_to_conc(R1eff, R1map, hp.acquisition.r1)

        idx, _ = np.where(np.isnan(C1))

        if idx.size != 0:
            C1[np.unique(idx)] = 1e-8

        C1 = np.reshape(C1, (-1, original_size[1], original_size[2], original_size[3]))

        hct = np.repeat(Hct_data[patient_dir], C1.shape[0])

        if first_patient:
            C1s = C1
            hcts = hct
            first_patient = False

        else:
            C1s = np.concatenate((C1s, C1), axis=0)
            hcts = np.concatenate((hcts, hct), axis=0)

        print(C1s.shape[0])

pickle.dump(slices_per_patient, open('mask_to_slice_'+args.dataset+'.p', "wb"))

save_name = 'all_patient_data_'+args.dataset+'.p'

if args.ip:
    save_name = save_name.replace('.p', '_ip.p')

pickle.dump([C1s, hcts], open(save_name, "wb"))
