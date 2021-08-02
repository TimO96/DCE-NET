import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import DCE_matt as matt
import hyperparams
import torch
import model_3D
import model
import simulations as sim
import pickle
import argparse
import time
from agreement import mean_diff_plot
import matplotlib
matplotlib.use('TkAgg')

parser = argparse.ArgumentParser()

parser.add_argument('--nn', type=str, default='linear')
parser.add_argument('--layers', type=int, nargs='+', default=[110, 110, 110])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dual_path', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--weights', action='store_true', default=False)
parser.add_argument('--multi_dim', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()

if args.nn in ['lstm', 'gru', 'convgru']:
    hp.network.attention = True

hp.network.nn = args.nn
hp.network.layers = args.layers
hp.training.lr = args.lr
hp.training.batch_size = args.batch_size
hp.training.optim = args.optim
hp.network.weighted_loss = args.weights
hp.network.dual_path = args.dual_path
hp.network.bidirectional = args.bidirectional

if args.multi_dim:
    network = '{}_{}_{}_{}_{}'.format(hp.network.nn,
                                      hp.network.layers,
                                      hp.training.lr,
                                      hp.training.batch_size,
                                      hp.network.dual_path)

else:
    network = '{}_{}_{}_{}'.format(hp.network.nn,
                                   hp.network.layers,
                                   hp.training.lr,
                                   hp.training.batch_size)

file = 'pretrained/pretrained_patient_data_'+network+'.pt'
file_params_mipa = 'params/patient_data_'+network+'_MIPA.p'
file_params_repro = 'params/patient_data_'+network+'_REPRO.p'

mask_to_slices_mipa = pickle.load(open('mask_to_slice_MIPA.p', "rb"))
mask_to_slices_repro = pickle.load(open('mask_to_slice_REPRO.p', "rb"))

if os.path.exists(file_params_mipa):
    params_mipa = pickle.load(open(file_params_mipa, 'rb'))

else:
    C1s, hcts = pickle.load(open('all_patient_data_MIPA_ip.p', "rb"))
    AIF = np.loadtxt('AIF.txt')
    hp.acquisition.timing = np.arange(160) * hp.simulations.time / 60
    aif = matt.fit_aif(AIF, hp.acquisition.timing)
    hp.aif.aif = aif

    hp.acquisition.timing = torch.FloatTensor(hp.acquisition.timing).to(hp.device)
    hp.acquisition.FAlist = [hp.acquisition.FA2]

    if args.multi_dim:
        hp.training.val_batch_size = 8
        net = model_3D.DCE_NET(hp).to(hp.device)
        net.load_state_dict(torch.load(file))
        begin = time.time()
        params = sim.predict_DCE(C1s, net, hp, Hct=hcts, one_dim=False)
        end = time.time()
        print('{} seconds'.format(end-begin))

    else:
        hp.training.val_batch_size = 4*160*160
        hcts = np.repeat(hcts, C1s.shape[1]*C1s.shape[2])
        C1s_reshaped = np.reshape(C1s, (-1, C1s.shape[-1]))
        net = model.DCE_NET(hp).to(hp.device)
        net.load_state_dict(torch.load(file))
        begin = time.time()
        params = sim.predict_DCE(C1s_reshaped, net, hp, Hct=hcts)
        end = time.time()
        print('{} seconds'.format(end-begin))

    pickle.dump(params[:4], open(file_params_mipa, 'wb'))
    params_mipa = params[:4]

if os.path.exists(file_params_repro):
    params_repro = pickle.load(open(file_params_repro, 'rb'))

else:
    C1s, hcts = pickle.load(open('all_patient_data_REPRO_ip.p', "rb"))
    AIF = np.loadtxt('AIF.txt')
    hp.acquisition.timing = np.arange(160) * hp.simulations.time / 60
    aif = matt.fit_aif(AIF, hp.acquisition.timing)
    hp.aif.aif = aif

    hp.acquisition.timing = torch.FloatTensor(hp.acquisition.timing).to(hp.device)
    hp.acquisition.FAlist = [hp.acquisition.FA2]

    if args.multi_dim:
        hp.training.val_batch_size = 8
        net = model_3D.DCE_NET(hp).to(hp.device)
        net.load_state_dict(torch.load(file))
        params = sim.predict_DCE(C1s, net, hp, Hct=hcts, one_dim=False)

    else:
        hp.training.val_batch_size = 4*160*160
        hcts = np.repeat(hcts, C1s.shape[1]*C1s.shape[2])
        C1s_reshaped = np.reshape(C1s, (-1, C1s.shape[-1]))
        net = model.DCE_NET(hp).to(hp.device)
        net.load_state_dict(torch.load(file))
        params = sim.predict_DCE(C1s_reshaped, net, hp, Hct=hcts)

    pickle.dump(params[:4], open(file_params_repro, 'wb'))
    params_repro = params[:4]

ke = np.reshape(params_mipa[0], (-1, 160, 160))
ve = np.reshape(params_mipa[2], (-1, 160, 160))
vp = np.reshape(params_mipa[3], (-1, 160, 160))

param_lsq = pickle.load(open('params/param_maps_patients_lsq_MIPA.p', "rb"))

ke_masked_means = [[], []]
ve_masked_means = [[], []]
vp_masked_means = [[], []]
ke_masked_means_lsq = [[], []]
ve_masked_means_lsq = [[], []]
vp_masked_means_lsq = [[], []]

count = 0

for patient in ['01', '02', '03', '04', '06', '11', '12', '14', '15']:
    for i in range(1, 3):
        patient_ke_masked_means = np.array([])
        patient_ve_masked_means = np.array([])
        patient_vp_masked_means = np.array([])
        patient_ke_masked_means_lsq = np.array([])
        patient_ve_masked_means_lsq = np.array([])
        patient_vp_masked_means_lsq = np.array([])
        patient_num = 'CR'+patient+'_'+str(i)

        if i == 1:
            path_mask = '/home/tottens/scratch/DCE_MRI/data/'+patient_num+'/kEp-label.nii.gz'
        else:
            path_mask = '/home/tottens/scratch/DCE_MRI/data/'+patient_num+'/kEp_1-label.nii.gz'

        mask = np.moveaxis(nib.load(path_mask).get_fdata(), 2, 0)

        if patient_num == 'CR01_1':
            mask = mask[1:25]
        elif patient_num == 'CR04_2':
            mask = mask[:29]
        else:
            mask = mask[1:29]

        slices = mask_to_slices_mipa[patient_num]

        for j, slice in enumerate(slices):
            if (mask[j] > 0).any():
                patient_ke_masked_means = np.append(patient_ke_masked_means,
                                                    (ke[slice]*mask[j])[np.nonzero(ke[slice]*mask[j])])
                patient_ve_masked_means = np.append(patient_ve_masked_means,
                                                    (ve[slice]*mask[j])[np.nonzero(ve[slice]*mask[j])])
                patient_vp_masked_means = np.append(patient_vp_masked_means,
                                                    (vp[slice]*mask[j])[np.nonzero(vp[slice]*mask[j])])

                patient_ke_masked_means_lsq = np.append(patient_ke_masked_means_lsq,
                                                        (param_lsq[slice, 0]*mask[j])[np.nonzero(param_lsq[slice, 0]*mask[j])])
                patient_ve_masked_means_lsq = np.append(patient_ve_masked_means_lsq,
                                                        (param_lsq[slice, 2]*mask[j])[np.nonzero(param_lsq[slice, 2]*mask[j])])
                patient_vp_masked_means_lsq = np.append(patient_vp_masked_means_lsq,
                                                        (param_lsq[slice, 3]*mask[j])[np.nonzero(param_lsq[slice, 3]*mask[j])])

        ke_masked_means[i-1].append(np.mean(patient_ke_masked_means))
        ve_masked_means[i-1].append(np.mean(patient_ve_masked_means))
        vp_masked_means[i-1].append(np.mean(patient_vp_masked_means))

        ke_masked_means_lsq[i-1].append(np.mean(patient_ke_masked_means_lsq))
        ve_masked_means_lsq[i-1].append(np.mean(patient_ve_masked_means_lsq))
        vp_masked_means_lsq[i-1].append(np.mean(patient_vp_masked_means_lsq))

ke = np.reshape(params_repro[0], (-1, 160, 160))
ve = np.reshape(params_repro[2], (-1, 160, 160))
vp = np.reshape(params_repro[3], (-1, 160, 160))

param_lsq = pickle.load(open('params/param_maps_patients_lsq_REPRO.p', "rb"))

for patient in range(1, 11):
    for i in range(1, 3):
        patient_ke_masked_means = np.array([])
        patient_ve_masked_means = np.array([])
        patient_vp_masked_means = np.array([])
        patient_ke_masked_means_lsq = np.array([])
        patient_ve_masked_means_lsq = np.array([])
        patient_vp_masked_means_lsq = np.array([])
        patient_num = 'REMP{:02d}_{}'.format(patient, i)
        if i == 1:
            path_mask = '/home/tottens/scratch/DCE_MRI/data/'+patient_num+'/kEp-label.nii.gz'
        else:
            path_mask = '/home/tottens/scratch/DCE_MRI/data/'+patient_num+'/kEp_1-label.nii.gz'

        mask = np.moveaxis(nib.load(path_mask).get_fdata(), 2, 0) == 1

        if patient_num == 'REMP08_2':
            mask = mask[:26]
        elif patient_num != 'REMP04_1':
            mask = mask[1:29]

        slices = mask_to_slices_repro[patient_num]

        for j, slice in enumerate(slices):
            if (mask[j] > 0).any():
                patient_ke_masked_means = np.append(patient_ke_masked_means,
                                                    (ke[slice]*mask[j])[np.nonzero(ke[slice]*mask[j])])
                patient_ve_masked_means = np.append(patient_ve_masked_means,
                                                    (ve[slice]*mask[j])[np.nonzero(ve[slice]*mask[j])])
                patient_vp_masked_means = np.append(patient_vp_masked_means,
                                                    (vp[slice]*mask[j])[np.nonzero(vp[slice]*mask[j])])

                patient_ke_masked_means_lsq = np.append(patient_ke_masked_means_lsq,
                                                        (param_lsq[slice, 0]*mask[j])[np.nonzero(param_lsq[slice, 0]*mask[j])])
                patient_ve_masked_means_lsq = np.append(patient_ve_masked_means_lsq,
                                                        (param_lsq[slice, 2]*mask[j])[np.nonzero(param_lsq[slice, 2]*mask[j])])
                patient_vp_masked_means_lsq = np.append(patient_vp_masked_means_lsq,
                                                        (param_lsq[slice, 3]*mask[j])[np.nonzero(param_lsq[slice, 3]*mask[j])])

        ke_masked_means[i-1].append(np.mean(patient_ke_masked_means))
        ve_masked_means[i-1].append(np.mean(patient_ve_masked_means))
        vp_masked_means[i-1].append(np.mean(patient_vp_masked_means))

        ke_masked_means_lsq[i-1].append(np.mean(patient_ke_masked_means_lsq))
        ve_masked_means_lsq[i-1].append(np.mean(patient_ve_masked_means_lsq))
        vp_masked_means_lsq[i-1].append(np.mean(patient_vp_masked_means_lsq))


fig, ax = plt.subplots(1, 3, figsize=(10, 3))
mean_diff_plot(np.array(ke_masked_means[0]), np.array(ke_masked_means[1]),
               xbound=(0, 1.1), ybound=0.5, decimal=2, label='$k_{ep}$ (min$^{-1}$)', ax=ax[0])
mean_diff_plot(np.array(ve_masked_means[0]), np.array(ve_masked_means[1]),
               xbound=(0, 0.8), ybound=0.6, decimal=2, label='$v_{e}$ [%]', ax=ax[1])
mean_diff_plot(np.array(vp_masked_means[0]), np.array(vp_masked_means[1]),
               xbound=(0, 0.065), ybound=0.05, decimal=2, label='$v_{p}$ [%]', ax=ax[2])

fig.tight_layout()
ax[1].set_title(args.nn+'\n', fontsize=16)
plt.subplots_adjust(top=0.76)
plt.show()
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
mean_diff_plot(np.array(ke_masked_means_lsq[0]), np.array(ke_masked_means_lsq[1]),
               xbound=(0, 1.1), ybound=0.5, decimal=2, label='$k_{ep}$ (min$^{-1}$)', ax=ax[0])
mean_diff_plot(np.array(ve_masked_means_lsq[0]), np.array(ve_masked_means_lsq[1]),
               xbound=(0, 0.8), ybound=0.6, decimal=2, label='$v_{e}$ [%]', ax=ax[1])
mean_diff_plot(np.array(vp_masked_means_lsq[0]), np.array(vp_masked_means_lsq[1]),
               xbound=(0, 0.065), ybound=0.05, decimal=2, label='$v_{p}$ [%]', ax=ax[2])

fig.tight_layout()
ax[1].set_title('NLLS\n', fontsize=16)
plt.subplots_adjust(top=0.76)
plt.show()
plt.close()
