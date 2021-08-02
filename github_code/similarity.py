import os
import numpy as np
import DCE_matt as matt
import hyperparams
import torch
import torch.nn as nn
import model_3D
import model
import simulations as sim
import pickle
import argparse
import ssim
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--nn', type=str, default='linear')
parser.add_argument('--layers', type=int, nargs='+', default=[110, 110, 110])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dual_path', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--weights', action='store_true', default=False)
parser.add_argument('--multi_dim', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()
file_data = 'all_patient_data_MIPA_ip.p'
C1s, hcts = pickle.load(open(file_data, "rb"))

file_results = 'patient_results.csv'

if os.path.exists(file_results):
    df = pd.read_csv(file_results)

else:
    df = pd.DataFrame({'Network': [], 'SSIM': [], 'nRMSE': []})

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
    network = '{}_{}_{}_{}.pt'.format(hp.network.nn,
                                      hp.network.layers,
                                      hp.training.lr,
                                      hp.training.batch_size)

file = 'pretrained/pretrained_patient_data_'+network+'.pt'

file_recons = 'recons/patient_data_'+network+'.pt'

C1s, hcts = pickle.load(open('all_patient_data_CR_3D_ip.p', "rb"))

if os.path.exists(file_recons):
    recons = pickle.load(open(file_recons, 'rb'))

else:
    if args.nn == 'lsq':
        C1s_recon = pickle.load(open('pred_curves_lsq.p', 'rb'))

    else:
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
            _, _, _, _, recons = sim.predict_DCE(C1s, net, hp, Hct=hcts, one_dim=False)

        else:
            hp.training.val_batch_size = 4*160*160
            hcts = np.repeat(hcts, C1s.shape[1]*C1s.shape[2])
            C1s_reshaped = np.reshape(C1s, (-1, C1s.shape[-1]))
            net = model.DCE_NET(hp).to(hp.device)
            net.load_state_dict(torch.load(file))
            _, _, _, _, recons = sim.predict_DCE(C1s_reshaped, net, hp, Hct=hcts)

        pickle.dump(recons, open(file_recons, 'wb'))

if args.multi_dim:
    C1s_recon = np.moveaxis(recons.astype(np.float32), 1, 3)

elif args.nn == 'lsq':
    C1s_recon = np.moveaxis(recons, 1, 3)

else:
    C1s_recon = np.reshape(recons, C1s.shape)

bad_points = np.mean(C1s, axis=3) < 1e-3
C1s_recon[bad_points] = C1s[bad_points]

C1s = torch.from_numpy(np.moveaxis(C1s.astype(np.float32), 3, 1))
C1s_recon = torch.from_numpy(np.moveaxis(C1s_recon.astype(np.float32), 3, 1))

ssim_loss = ssim.SSIM()
mse_loss_full = nn.MSELoss(reduction='none')
mse_loss = nn.MSELoss()

similarity = 0
nrmse = 0
rmse = 0

for i in tqdm(range(len(C1s))):
    similarity += ssim_loss(C1s[i].unsqueeze(0), C1s_recon[i].unsqueeze(0)).item()
    nrmse_part = torch.mean(torch.sqrt(mse_loss_full(C1s[i], C1s_recon[i])), dim=(1, 2))
    nrmse_diff = torch.amax(C1s[i], dim=(1, 2))-torch.amin(C1s[i], dim=(1, 2))
    nrmse += torch.mean(torch.div(nrmse_part, nrmse_diff)).item()
    rmse += torch.sqrt(mse_loss(C1s[i], C1s_recon[i])).item()

print('network: {} ; SSIM: {} ; nRMSE: {} ; RMSE: {}'.format(network,
                                                             similarity/len(C1s),
                                                             nrmse/len(C1s),
                                                             rmse/len(C1s)))

df = df.append({'Network': network, 'SSIM': similarity/len(C1s),
                'nRMSE': nrmse/len(C1s), 'RMSE': rmse/len(C1s)}, ignore_index=True)

df.to_csv(file_results, index=False)
