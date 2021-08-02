import hyperparams
import argparse
import numpy as np
import simulations as sim
import os

# np.random.seed(42)
# torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--nn', type=str, default='linear')
parser.add_argument('--layers', type=int, nargs='+', default=[160, 160, 160])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--supervised', action='store_true', default=False)
parser.add_argument('--results', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()

hp.training.lr = args.lr
hp.training.batch_size = args.batch_size
hp.network.nn = args.nn
hp.network.layers = args.layers
hp.network.attention = args.attention
hp.network.bidirectional = args.bidirectional
hp.supervised = args.supervised

# create save name for framework
hp.exp_name = ''
arg_dict = vars(args)
for i, arg in enumerate(arg_dict):
    if i == len(arg_dict)-2:
        hp.exp_name += str(arg_dict[arg])
        break
    else:
        hp.exp_name += '{}_'.format(arg_dict[arg])

print(hp.exp_name)

if os.path.exists(hp.create_name):
    hp.create_data = False
else:
    hp.create_data = True

# execute the non-linear least squares method
if hp.network.nn == 'lsq':
    hp.simulations.num_samples_leval = 10000
    SNRs = [7, 10, 20, 30, 40, 60, 80, 100]
    params = np.zeros((len(SNRs), 4, 2))
    for i, SNR in enumerate(SNRs):
        file = hp.create_name.replace('.p', '')+'_'+str(SNR)+'.p'
        if os.path.exists(file):
            hp.create_data = False
        else:
            hp.create_data = True

        params[i] = sim.run_simulations(hp, SNR=SNR, eval=True)

    np.save('results'+hp.exp_name+'.npy', params)

# train a neural network based approach
elif not args.results:
    sim.run_simulations(hp, SNR='all')

# passing --results will perform evaluation on the given framework
else:
    var_seq = False  # option for different acquisition lengths

    hp.simulations.num_samples = 10000
    SNRs = [7, 10, 20, 30, 40, 60, 80, 100]
    datapoints = [80, 90, 100, 110, 120, 130, 140, 150, 160]

    params = np.zeros((len(datapoints), 4, 2))
    if var_seq:
        for i, datapoint in enumerate(datapoints):
            hp.acquisition.rep2 = datapoint
            hp.simulations.Tonset_min = hp.simulations.time * datapoint//6
            hp.simulations.Tonset_max = hp.simulations.time * datapoint//5

            file = hp.create_name.replace('.p', '')+'seq_'+str(hp.acquisition.rep2)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            params[i] = sim.run_simulations(hp, SNR=20, eval=True, var_seq=True)

        np.save('results'+hp.exp_name+'_var_seq.npy', params)

    else:
        params = np.zeros((len(SNRs), 4, 2))
        for i, SNR in enumerate(SNRs):
            file = hp.create_name.replace('.p', '')+'_'+str(SNR)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            params[i] = sim.run_simulations(hp, SNR=SNR, eval=True)

        np.save('results'+hp.exp_name+'.npy', params)
