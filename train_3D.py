# import libraries
import torch
import torch.nn as nn
import torch.utils.data as utils
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import model_3D as model
import matplotlib
matplotlib.use('TkAgg')

# np.random.seed(42)
# torch.manual_seed(42)


def NMSE_loss(input, target, weight=None, reduction='mean'):
    if weight is not None:
        all_losses = weight * ((input-target) ** 2) / (target**2)
        if reduction == 'mean':
            loss = torch.sum(all_losses) / torch.sum(weight)
        elif reduction == 'eval':
            loss = torch.mean(all_losses, axis=1)
        else:
            raise ValueError('not a valid reduction')

    else:
        all_losses = ((input-target) ** 2) / (target**2)
        if reduction == 'mean':
            loss = torch.mean(all_losses)
        elif reduction == 'eval':
            loss = torch.mean(all_losses, axis=1)
        else:
            raise ValueError('not a valid reduction')

    return loss


def normalize_params(pred_params, orig_params, bounds):
    pred_params = pred_params.T
    for i in range(len(bounds)):
        pred_params[:, i] /= (bounds[1, i] - bounds[0, i])
        orig_params[:, i] /= (bounds[1, i] - bounds[0, i])

    return pred_params, orig_params


def train(C1, hp, net=None, Hct=None, orig_params=None):
    if hp.use_cuda:
        torch.backends.cudnn.benchmark = True

    if net is None:
        net = model.DCE_NET(copy.deepcopy(hp)).to(hp.device)

    Hct = np.expand_dims(Hct, axis=(1, 2, 3))
    Hct = np.repeat(np.repeat(Hct, C1.shape[1], axis=1), C1.shape[2], axis=2)
    C1 = np.concatenate([Hct, C1], axis=3)
    C1 = np.moveaxis(C1, 3, 1)

    # Loss function and optimizer
    criterion = nn.MSELoss().to(hp.device)

    # Data loader
    split = int(np.floor(len(C1)*hp.training.split))
    C1 = torch.from_numpy(C1.astype(np.float32))

    train_set, val_set = torch.utils.data.random_split(C1, [split, len(C1)-split])
    trainloader = utils.DataLoader(train_set,
                                   batch_size=hp.training.batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   drop_last=True)

    valloader = utils.DataLoader(val_set,
                                 batch_size=hp.training.val_batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=True)

    num_batches = len(train_set) // hp.training.batch_size

    num_batches2 = len(val_set) // hp.training.val_batch_size

    if num_batches > hp.training.totalit:
        totalit = hp.training.totalit
    else:
        totalit = num_batches

    if not os.path.exists(hp.out_fold):
        os.makedirs(hp.out_fold)

    optimizer, scheduler = model.load_optimizer(net, hp)

    params_total = sum(p.numel() for p in net.parameters())
    train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # fix for sudden nan values in patient data
    for name, p in net.named_parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.nan_to_num(grad))

    print(params_total, 'params in total')
    print(train_params, 'trainable params in total')

    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    bound = hp.max_rep+1

    for epoch in range(hp.training.epochs):
        print("-----------------------------------------------------------------")
        print("\nEpoch:{}; Current best val_loss:{}".format(epoch, best))

        train_loss = 0.
        val_loss = 0.
        hp.acquisition.timing = hp.acquisition.timing.to(hp.device)

        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i == totalit:
                break

            X_batch = X_batch.to(hp.device)

            optimizer.zero_grad()

            X_pred, ke, dt, ve, vp = net(X_batch[:, 1:bound], Hct=X_batch[:, 0])

            loss = criterion(X_pred[:, :bound], X_batch[:, 1:bound])

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # evaluation
        with torch.no_grad():
            for i, X_batch in enumerate(tqdm(valloader, position=0, leave=True), 0):
                X_batch = X_batch.to(hp.device)

                optimizer.zero_grad()

                X_pred, ke, dt, ve, vp = net(X_batch[:, 1:bound], Hct=X_batch[:, 0])

                loss = criterion(X_pred[:, :bound], X_batch[:, 1:bound])

                val_loss += loss.item()

        # scale losses
        train_loss = train_loss/totalit*1000
        val_loss = val_loss/num_batches2*1000
        loss_train.append(train_loss)
        loss_val.append(val_loss)

        if hp.training.optim_patience > 0:
            scheduler.step(val_loss)

        # early stopping
        if val_loss < best:
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = val_loss
            num_bad_epochs = 0

            print("\nLoss: {}; val_loss: {}; bad epochs: {}".format(train_loss,
                                                                    val_loss,
                                                                    num_bad_epochs))

        else:
            num_bad_epochs += 1

            print("\nLoss: {}; val_loss: {}; bad epochs: {}".format(train_loss,
                                                                    val_loss,
                                                                    num_bad_epochs))

            # early stopping
            if num_bad_epochs == hp.training.patience:
                print("\nEarly stopping, best val loss: {}".format(best))
                print("Done with DCE fitting")
                break

        # calculate the best, median and worst fits based on NMSE loss
        X_pred = torch.moveaxis(X_pred, 1, 3).reshape(-1, X_pred.size(1))
        X_batch = torch.moveaxis(X_batch[:, 1:], 1, 3).reshape(-1, X_pred.size(-1))
        all_losses = NMSE_loss(X_pred, X_batch, reduction='eval')

        values_top, inds_top = torch.topk(all_losses, int(len(all_losses)/2))
        values_bottom, inds_bottom = torch.topk(all_losses, 2, largest=False)
        values = torch.cat((values_top[:2], values_top[-2:], values_bottom))
        inds = torch.cat((inds_top[:2], inds_top[-2:], inds_bottom))

        minmax_batch_losses = X_batch[inds].cpu()
        minmax_pred_losses = X_pred[inds].cpu()

        do_plots(hp, epoch, minmax_batch_losses, minmax_pred_losses, loss_train, loss_val, values, name='dce_part')

        # do_plots_3D(hp, epoch, X_batch, X_pred, ke, ve, vp, loss_train, loss_val)

    print("Done")
    net.load_state_dict(final_model)

    return net


def do_plots(hp, epoch, X_batch, X_pred, loss_train, loss_val, values, loss_train_curve=None, loss_val_curve=None, name=None):
    # plot loss history
    hp.acquisition.timing = hp.acquisition.timing.cpu()
    plt.close('all')

    labels = ['worst', 'median', 'best']
    fig, axs = plt.subplots(int(len(values)/2)+1, 2, figsize=(6,5))

    for i in range(len(values)):
        axs[int(i/2), i%2].plot(hp.acquisition.timing, X_batch.data[i])
        axs[int(i/2), i%2].plot(hp.acquisition.timing, X_pred.data[i])
        axs[int(i/2), i%2].set_title('{} {}, loss:{:.2e}'.format(labels[int(i/2)], (i%2)+1, values[i].item()))

    for ax in axs.flat:
        ax.set(xlabel='time (m)', ylabel='signal (a.u.)')

    for ax in axs.flat:
        ax.label_outer()

    axs[3, 0].plot(loss_train)
    axs[3, 0].plot(loss_val)
    axs[3, 0].set_yscale('log')
    axs[3, 0].set_xlabel('epoch')
    axs[3, 0].set_ylabel('loss')

    plt.ion()
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

    if hp.training.save_train_fig:
        plt.gcf()
        plt.savefig('{out_fold}/{name}_fit_{epoch}.png'.format(out_fold=hp.out_fold, epoch=epoch, name=name))

    return fig


def do_plots_3D(hp, epoch, X_batch, X_pred, ke, ve, vp, loss_train, loss_val):
    # plot loss history
    hp.acquisition.timing = hp.acquisition.timing.cpu()
    plt.close('all')
    if hp.supervised or hp.network.aif:
        X_batch = X_batch[:, 1:]

    fig, axs = plt.subplots(2, 2)

    ax1 = axs[0, 0].imshow(ke[0].cpu().numpy().T, cmap='jet')
    axs[0, 0].set_title('kep')
    ax2 = axs[0, 1].imshow(ve[0].cpu().numpy().T, cmap='jet')
    axs[0, 1].set_title('ve')
    ax3 = axs[1, 0].imshow(vp[0].cpu().numpy().T, cmap='jet')
    axs[1, 0].set_title('vp')
    fig.colorbar(ax1, ax=axs[0, 0])
    fig.colorbar(ax2, ax=axs[0, 1])
    fig.colorbar(ax3, ax=axs[1, 0])

    axs[1, 1].plot(loss_train)
    axs[1, 1].plot(loss_val)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel('epoch')
    axs[1, 1].set_ylabel('loss')

    plt.ion()
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

    if hp.training.save_train_fig:
        plt.gcf()
        plt.savefig('{out_fold}/{name}_fit_{epoch}.png'.format(out_fold=hp.out_fold, epoch=epoch, name='dce-part'))
