import os
import numpy as np
import DCE_matt as matt
import hyperparams
import train_3D
import model_3D
import train
import model
import torch
import pickle
import argparse
from tqdm import tqdm
import time

# np.random.seed(42)
# torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--nn', type=str, default='convlin')
parser.add_argument('--layers', type=int, nargs='+', default=[110, 110, 110])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--dual_path', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--cpu', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()
spatiotemporal = False
file_data = 'all_patient_data_MIPA_ip.p'
C1s, hcts = pickle.load(open(file_data, "rb"))
print('dataset loaded')

if args.nn in ['lstm', 'gru', 'convgru']:
    hp.network.attention = True

hp.network.nn = args.nn
hp.network.layers = args.layers
hp.training.lr = args.lr
hp.training.optim = args.optim
hp.training.batch_size = args.batch_size
hp.training.val_batch_size = args.batch_size*8
hp.network.dual_path = args.dual_path
hp.network.bidirectional = args.bidirectional

if args.cpu:
    hp.device = torch.device('cpu')

if hp.network.nn in ['convlin', 'convgru', 'unet']:
    spatiotemporal = True

# Compute parameters of AIF according to Cosine8AIF
AIF = np.loadtxt('AIF.txt')
hp.acquisition.timing = np.arange(160) * hp.simulations.time / 60
aif = matt.fit_aif(AIF, hp.acquisition.timing)
hp.aif.aif = aif

# hcts = np.repeat(0.4, len(C1s))


def slice_images(C1s, hcts):
    sliced_C1s = np.zeros((len(C1s)*1600, C1s.shape[1]//40, C1s.shape[2]//40, C1s.shape[3]))
    sliced_hcts = np.zeros((len(hcts)*1600))

    for i in tqdm(range(len(C1s))):
        for j in range(40):
            for k in range(40):
                sliced_C1s[(i*1600)+(j*40)+k] = C1s[i, j*4:(j+1)*4, k*4:(k+1)*4]
                sliced_hcts[(i*1600)+(j*40)+k] = hcts[i]

    pickle.dump([sliced_C1s, sliced_hcts], open('all_patient_data_sliced_ip.p', "wb"))

    print('Done')

    return (sliced_C1s, sliced_hcts)


if spatiotemporal:
    hp.training.val_batch_size = 8
    hp.acquisition.timing = torch.FloatTensor(hp.acquisition.timing).to(hp.device)
    hp.acquisition.FAlist = [hp.acquisition.FA2]
    C1s[np.mean(C1s, axis=3) < 1e-2] = 1e-8

    C1s[C1s > 15] = 15
    C1s[C1s < -5] = -5
    #C1s[np.std(C1s, axis=3) <= 1e-3] = 1e-8

    if args.pretrained:
        net = model_3D.DCE_NET(hp).to(hp.device)
        net.load_state_dict(torch.load('pretrained/pretrained_patient_data_{}_{}_{}_{}_{}.pt'.format(hp.network.nn,
                                                                                                     hp.training.lr,
                                                                                                     hp.training.batch_size,
                                                                                                     hp.training.optim,
                                                                                                     hp.network.dual_path
                                                                                                     )))

        print('start training process')
        net = train_3D.train(C1s, hp, net=net, Hct=hcts)

    else:
        print('start training process')
        net = train_3D.train(C1s, hp, Hct=hcts)

    torch.save(net.state_dict(), 'pretrained/pretrained_patient_data_{}_{}_{}_{}_{}.pt'.format(hp.network.nn,
                                                                                               hp.network.layers,
                                                                                               hp.training.lr,
                                                                                               hp.training.batch_size,
                                                                                               hp.network.dual_path
                                                                                               ))

elif args.nn != 'lsq':
    hp.training.val_batch_size = 160*160
    hp.acquisition.timing = torch.FloatTensor(hp.acquisition.timing).to(hp.device)
    hp.acquisition.FAlist = [hp.acquisition.FA2]
    C1s[np.mean(C1s, axis=3) < 1e-2] = 1e-8
    hcts = np.repeat(hcts, C1s.shape[1]*C1s.shape[2])
    C1s = np.reshape(C1s, (-1, C1s.shape[-1]))

    C1s[C1s > 15] = 15
    C1s[C1s < -5] = -5

    mask = np.mean(C1s, axis=1) > 1e-3
    print('{} of {} points not used'.format(len(C1s)-len(C1s[mask]), len(C1s)))

    if args.pretrained:
        net = model.DCE_NET(hp).to(hp.device)
        net.load_state_dict(torch.load('pretrained/pretrained_patient_data_{}_{}_{}_{}_{}.pt'.format(hp.network.nn,
                                                                                                     hp.network.layers,
                                                                                                     hp.training.lr,
                                                                                                     hp.training.batch_size,
                                                                                                     hp.network.bidirectional
                                                                                                     )))

        print('start training process')
        net = train.train(C1s[mask], hp, net=net, Hct=hcts[mask])

    print('start training process')
    net = train.train(C1s[mask], hp, Hct=hcts[mask])

    torch.save(net.state_dict(), 'pretrained/pretrained_patient_data_{}_{}_{}_{}_{}.pt'.format(hp.network.nn,
                                                                                               hp.network.layers,
                                                                                               hp.training.lr,
                                                                                               hp.training.batch_size,
                                                                                               hp.network.bidirectional
                                                                                               ))

else:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    param_maps = np.zeros((C1s.shape[0], 4, C1s.shape[1], C1s.shape[2]))

    begin = time.time()
    for i in range(len(C1s)):
        aif = hp.aif.aif.copy()
        aif['ab'] /= (1-hcts[i])
        C1 = np.reshape(C1s[i], (-1, C1s[i].shape[-1]))
        out = matt.fit_tofts_model(C1, hp.acquisition.timing, aif, jobs=8)
        param_maps[i, 0] = np.reshape(out[0], (C1s.shape[1], C1s.shape[2]))
        param_maps[i, 1] = np.reshape(out[1], (C1s.shape[1], C1s.shape[2]))
        param_maps[i, 2] = np.reshape(out[2], (C1s.shape[1], C1s.shape[2]))
        param_maps[i, 3] = np.reshape(out[3], (C1s.shape[1], C1s.shape[2]))

        pickle.dump(param_maps, open('params/param_maps_patients_lsq.p', "wb"))

    end = time.time()
    print('{} seconds'.format(end-begin))
