import torch
import torch.utils.data as utils
import DCE_matt as dce
import pydcemri.dcemri as classic
import copy
import numpy as np
import model
import pickle
import time
import train
from tqdm import tqdm


def run_simulations(hp, SNR=15, eval=False, var_seq=False):
    print(hp.device)
    if hp.network.nn == 'lsq':
        hp.simulations.num_samples = hp.simulations.num_samples_leval

    rep1 = hp.acquisition.rep1-1
    rep2 = hp.acquisition.rep2-1

    if hp.create_data:
        # create simulation parameters
        test = np.random.uniform(0, 1, (hp.simulations.num_samples, 1))
        vp = hp.simulations.vp_min + (test * (hp.simulations.vp_max - hp.simulations.vp_min))
        test = np.random.uniform(0, 1, (hp.simulations.num_samples, 1))
        ve = hp.simulations.ve_min + (test * (hp.simulations.ve_max - hp.simulations.ve_min))
        test = np.random.uniform(0, 1, (hp.simulations.num_samples, 1))
        kep = hp.simulations.kep_min + (test * (hp.simulations.kep_max - hp.simulations.kep_min))

        Tonset = np.random.uniform(hp.simulations.Tonset_min, hp.simulations.Tonset_max, (hp.simulations.num_samples, 1))
        test = np.random.uniform(0, 1, (hp.simulations.num_samples, 1))
        R1 = hp.simulations.R1_min + (test * (hp.simulations.R1_max - hp.simulations.R1_min))

        hp.acquisition.FAlist = np.array([hp.acquisition.FA2])

        del test

        hp.acquisition.timing = np.arange(0, rep2 + rep1) * hp.simulations.time / 60

        num_time = len(hp.acquisition.timing)
        X_dw = np.zeros((hp.simulations.num_samples, num_time))
        R1eff = np.zeros((hp.simulations.num_samples, num_time))
        C = np.zeros((hp.simulations.num_samples, num_time))

        test = np.random.uniform(0, 1, (hp.simulations.num_samples))

        # vary the Hct value from 0.3 to 0.6
        if hp.network.aif:
            Hct = 0.3 + (test * (0.3))

        else:
            Hct = None
            aif = hp.aif.aif
            aif['ab'] /= (1-hp.aif.Hct)

        if hp.network.full_aif or eval:
            AIF_curves = np.zeros((len(Hct), rep2))

        for aa in tqdm(range(len(kep))):
            if hp.network.aif:
                aif = hp.aif.aif.copy()
                aif['ab'] /= (1-Hct[aa])

            C[aa, :] = dce.Cosine8AIF_ExtKety(hp.acquisition.timing, aif, kep[aa][0], (Tonset[aa][0] + rep1*hp.simulations.time)/60,
                                              ve[aa][0], vp[aa][0])

            R1eff[aa, :] = classic.con_to_R1eff(C[aa, :], R1[aa][0], hp.acquisition.r1)
            X_dw[aa, :] = classic.r1eff_to_dce(R1eff[aa, :], hp.acquisition.TR, hp.acquisition.FAlist)

            if hp.network.full_aif or eval:
                AIF_curves[aa] = dce.Cosine8AIF(hp.acquisition.timing, aif['ab'], aif['ar'], aif['ae'], aif['mb'], 
                                                aif['mm'], aif['mr'], aif['me'], aif['tr'], aif['t0'])

        # scale the signal to the baseline signal
        S0_out = np.mean(X_dw[:, :rep2//10], axis=1)
        dce_signal_scaled = X_dw / S0_out[:, None]

        # vary SNR from 7 to 100
        if SNR == 'all':
            SNR = np.linspace(7, 100, num=hp.simulations.num_samples)

        noise = np.random.normal(0, 1/SNR, (num_time, hp.simulations.num_samples)).T
        dce_signal_noisy = dce_signal_scaled + noise

        del noise, X_dw, R1eff, C

        hp.acquisition.timing = torch.FloatTensor(hp.acquisition.timing)

        # if not lsq, convert to concentration and save data
        if hp.network.nn != 'lsq':
            S0 = np.mean(dce_signal_noisy[:, :rep2//10], axis=1)
            R1eff2 = dce.dce_to_r1eff(dce_signal_noisy, S0, R1.squeeze(), hp.acquisition.TR, hp.acquisition.FA2)
            C1 = dce.r1eff_to_conc(R1eff2, R1, hp.acquisition.r1)

            dce_signal_noisy = C1
            if hp.network.full_aif or eval:
                data = [dce_signal_noisy, hp, Hct, kep, ve, vp, Tonset, AIF_curves]
            else:
                data = [dce_signal_noisy, hp, Hct, kep, ve, vp, Tonset]

            # change name when varying SNR or acquisition points
            if eval:
                if var_seq:
                    hp.create_name_copy = hp.create_name.replace('.p', '')+'seq_'+str(hp.acquisition.rep2)+'.p'
                else:
                    hp.create_name_copy = hp.create_name.replace('.p', '')+'_'+str(SNR)+'.p'
            else:
                hp.create_name_copy = hp.create_name

            pickle.dump(data, open(hp.create_name_copy, "wb"))

        hp.acquisition.timing = hp.acquisition.timing.to(hp.device)

    else:
        print('load simulation data')
        begin = time.time()

        if eval:
            if var_seq:
                hp.create_name_copy = hp.create_name.replace('.p', '')+'seq_'+str(hp.acquisition.rep2)+'.p'
            else:
                hp.create_name_copy = hp.create_name.replace('.p', '')+'_'+str(SNR)+'.p'
        else:
            hp.create_name_copy = hp.create_name

        if hp.network.full_aif or eval:
            dce_signal_noisy, hp_data, Hct, kep, ve, vp, Tonset, AIF_curves = pickle.load(open(hp.create_name_copy, "rb"))
        else:
            dce_signal_noisy, hp_data, Hct, kep, ve, vp, Tonset = pickle.load(open(hp.create_name_copy, "rb"))

        time_passed = time.time() - begin
        print('{} data points, loading time:{:2f} s'.format(len(dce_signal_noisy), time_passed))

        hp.acquisition = hp_data.acquisition
        hp.aif = hp_data.aif
        hp.acquisition.timing = hp.acquisition.timing.to(hp.device)

    # executing non-linear least squares fit on data
    if hp.network.nn == 'lsq':
        out = np.zeros((4, hp.simulations.num_samples_leval))

        for i in tqdm(range(hp.simulations.num_samples)):
            aif = hp.aif.aif.copy()
            aif['ab'] /= (1-Hct[i])

            params = dce.fit_tofts_model(np.expand_dims(dce_signal_noisy[i], axis=0), hp.acquisition.timing.cpu().numpy(), aif,
                                         X0=(np.mean(kep), np.mean(Tonset)/60, np.mean(ve), np.mean(vp)))

            out[0, i] = params[0]
            out[1, i] = params[1]
            out[2, i] = params[2]
            out[3, i] = params[3]

    # loading model for evaluation on neural networks
    elif eval:
        net = model.DCE_NET(hp).to(hp.device)
        net.load_state_dict(torch.load('pretrained/pretrained_'+hp.exp_name+'.pt'))
        net.to(hp.device)
        if hp.network.full_aif:
            Hct = np.concatenate([Hct[:, np.newaxis], AIF_curves], axis=1)

    # start training for neural networks
    else:
        if hp.network.full_aif:
            Hct = np.concatenate([Hct[:, np.newaxis], AIF_curves], axis=1)

        if hp.pretrained:
            net = model.DCE_NET(hp).to(hp.device)
            net.load_state_dict(torch.load(hp.pretrain_name))
            net.to(hp.device)

            net = train.train(dce_signal_noisy, hp, net=net, Hct=Hct,
                              orig_params=torch.Tensor([np.squeeze(kep),
                                                        np.squeeze(ve),
                                                        np.squeeze(vp),
                                                        (np.squeeze(Tonset)+rep1*hp.simulations.time) / 60]))
        else:
            net = train.train(dce_signal_noisy, hp, Hct=Hct,
                              orig_params=torch.Tensor([np.squeeze(kep),
                                                        np.squeeze(ve),
                                                        np.squeeze(vp),
                                                        (np.squeeze(Tonset)+rep1*hp.simulations.time) / 60]))

        torch.save(net.state_dict(), 'pretrained/pretrained_'+hp.exp_name+'.pt')

    # evaluate on current dataset
    if hp.network.nn != 'lsq':
        out = predict_DCE(dce_signal_noisy, copy.deepcopy(net), hp, Hct=Hct)

    param_results = sim_results(out, hp, kep, ve, vp, Tonset)

    return param_results


def predict_DCE(C1, net, hp, Hct=None, one_dim=True):
    net.eval()

    first_params = True

    C1[np.isnan(C1)] = 0

    # perform interpolation for FCN when acquisition points is lower than max rep
    if hp.network.nn == 'linear' and C1.shape[1] < hp.max_rep:
        delta = (C1.shape[1]-1) / (hp.max_rep-1)
        C = np.zeros((C1.shape[0], hp.max_rep))
        for i in tqdm(range(len(C1))):
            C[i] = np.array([train.interpolate(C1[i], j*delta) for j in range(hp.max_rep)])

        C1 = C
        hp.acquisition.timing = torch.arange(hp.max_rep, device=hp.device) * hp.simulations.time / 60

    print('using full network with voxel-wise aif')

    # temporal framework
    if one_dim:
        if hp.network.full_aif:
            C1 = np.concatenate([Hct, C1], axis=1)
        else: 
            C1 = np.concatenate([Hct[:, np.newaxis], C1], axis=1)
        
        ke = torch.zeros(len(C1))
        ve = torch.zeros(len(C1))
        vp = torch.zeros(len(C1))
        dt = torch.zeros(len(C1))
        X = torch.zeros((len(C1), 160))

    # spatiotemporal framework
    else:
        Hct = np.expand_dims(Hct, axis=(1, 2, 3))
        Hct = np.repeat(np.repeat(Hct, C1.shape[1], axis=1), C1.shape[2], axis=2)
        C1 = np.concatenate([Hct, C1], axis=3)
        C1 = np.moveaxis(C1, 3, 1)

        ke = torch.zeros((C1.shape[0], C1.shape[2], C1.shape[3]))
        ve = torch.zeros((C1.shape[0], C1.shape[2], C1.shape[3]))
        vp = torch.zeros((C1.shape[0], C1.shape[2], C1.shape[3]))
        dt = torch.zeros((C1.shape[0], C1.shape[2], C1.shape[3]))
        X = torch.zeros((C1.shape[0], 160, C1.shape[2], C1.shape[3]))

    C1 = torch.from_numpy(C1.astype(np.float32))

    inferloader = utils.DataLoader(C1,
                                   batch_size=hp.training.val_batch_size,
                                   shuffle=False,
                                   drop_last=False)

    # perform inference
    size = hp.training.val_batch_size

    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.to(hp.device)

            if hp.network.full_aif:
                X_dw, ket, dtt, vet, vpt = net(X_batch[:, hp.acquisition.rep2:], Hct=X_batch[:, :hp.acquisition.rep2])
            else:
                X_dw, ket, dtt, vet, vpt = net(X_batch[:, 1:], Hct=X_batch[:, :1])

            ke[i*size:(i+1)*size] = ket.cpu().squeeze()
            ve[i*size:(i+1)*size] = vet.cpu().squeeze()
            vp[i*size:(i+1)*size] = vpt.cpu().squeeze()
            dt[i*size:(i+1)*size] = dtt.cpu().squeeze()
            X[i*size:(i+1)*size] = X_dw.cpu().squeeze()
            # if first_params:
            #     ke = ket.cpu().numpy()
            #     dt = dtt.cpu().numpy()
            #     ve = vet.cpu().numpy()
            #     vp = vpt.cpu().numpy()
            #     X = X_dw.cpu().numpy()
            #     first_params = False

            # else:
            #     ke = np.concatenate((ke, ket.cpu().numpy()), axis=0)
            #     dt = np.concatenate((dt, dtt.cpu().numpy()), axis=0)
            #     ve = np.concatenate((ve, vet.cpu().numpy()), axis=0)
            #     vp = np.concatenate((vp, vpt.cpu().numpy()), axis=0)
            #     X = np.concatenate((X, X_dw.cpu().numpy()), axis=0)

    ke = np.array(ke)
    ve = np.array(ve)
    vp = np.array(vp)
    dt = np.array(dt)
    X = np.array(X)

    params = [ke, dt, ve, vp, X]

    return params


def sim_results(paramsNN_full, hp, kep, ve, vp, Tonset, Hct=None):
    # calculate the random and systematic error of every parameter
    rep1 = hp.acquisition.rep1 - 1
    error_ke = paramsNN_full[0] - np.squeeze(kep)
    randerror_ke = np.std(error_ke)
    syserror_ke = np.mean(error_ke)
    del error_ke

    error_ve = paramsNN_full[2] - np.squeeze(ve)
    randerror_ve = np.std(error_ve)
    syserror_ve = np.mean(error_ve)
    del error_ve

    error_vp = paramsNN_full[3] - np.squeeze(vp)
    randerror_vp = np.std(error_vp)
    syserror_vp = np.mean(error_vp)
    del error_vp

    error_dt = paramsNN_full[1] - (np.squeeze(Tonset) + rep1 * hp.simulations.time) / 60
    randerror_dt = np.std(error_dt)
    syserror_dt = np.mean(error_dt)
    del error_dt

    normke = np.mean(kep)
    normve = np.mean(ve)
    normvp = np.mean(vp)
    normdt = np.mean(Tonset / 60)
    print('ke_sim, dke_lsq, sys_ke_lsq, dke, sys_ke')
    print([normke, '  ', randerror_ke, '  ', syserror_ke])
    print([normve, '  ', randerror_ve, '  ', syserror_ve])
    print([normvp, '  ', randerror_vp, '  ', syserror_vp])
    print([normdt, '  ', randerror_dt, '  ', syserror_dt])

    # return np.array([[randerror_ke, syserror_ke],
    #                  [randerror_ve, syserror_ve],
    #                  [randerror_vp, syserror_vp],
    #                  [randerror_dt, syserror_dt]])

    return np.array(paramsNN_full[:4])
