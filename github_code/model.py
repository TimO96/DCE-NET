# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import functions
import matplotlib
matplotlib.use('TkAgg')

# np.random.seed(42)
# torch.manual_seed(42)


class DCE_NET(nn.Module):
    def __init__(self, hp):
        super(DCE_NET, self).__init__()
        self.hp = hp

        netlen_base = hp.max_rep
        self.hp.acquisition.FAlistnet = torch.FloatTensor(np.expand_dims(self.hp.acquisition.FAlist, axis=1)).to(hp.device)

        if self.hp.network.nn == 'linear':
            input_size = netlen_base

            self.encoder_layers = nn.ModuleList()

            for i, hidden_neurons in enumerate(self.hp.network.layers):
                self.encoder_layers.extend([nn.Linear(input_size, hidden_neurons),
                                            #nn.BatchNorm1d(hidden_neurons),
                                            nn.ELU(),
                                            #nn.Dropout(self.hp.network.dropout)
                                            ])

                input_size = hidden_neurons

            self.linear = nn.Sequential(*self.encoder_layers)

            self.encoder = nn.Sequential(nn.Linear(self.hp.network.layers[-1]+1, int((self.hp.network.layers[-1]+1)/2)),
                                         #nn.BatchNorm1d(int((self.hp.network.layers[-1]+1)/2)),
                                         nn.ELU(),
                                         nn.Linear(int((self.hp.network.layers[-1]+1)/2), 4)
                                         )

        elif self.hp.network.nn in ['lstm', 'gru']:
            if self.hp.network.nn == 'lstm':
                if self.hp.network.bidirectional:
                    self.rnn = nn.LSTM(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True, bidirectional=True)
                    hidden_dim = self.hp.network.layers[0]*2

                else:
                    self.rnn = nn.LSTM(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True)
                    hidden_dim = self.hp.network.layers[0]

            else:
                if self.hp.network.bidirectional:
                    self.rnn = nn.GRU(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True, bidirectional=True)
                    hidden_dim = self.hp.network.layers[0]*2

                else:
                    self.rnn = nn.GRU(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True)
                    hidden_dim = self.hp.network.layers[0]

            if self.hp.network.attention:
                self.score_ke = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
                self.score_ve = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
                self.score_vp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
                self.score_dt = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

                self.encoder_ke = nn.Sequential(nn.Linear(hidden_dim+1, int((hidden_dim+1)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim+1)/2), 1)
                                                )
                self.encoder_ve = nn.Sequential(nn.Linear(hidden_dim+1, int((hidden_dim+1)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim+1)/2), 1)
                                                )
                self.encoder_vp = nn.Sequential(nn.Linear(hidden_dim+1, int((hidden_dim+1)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim+1)/2), 1)
                                                )
                self.encoder_dt = nn.Sequential(nn.Linear(hidden_dim+1, int((hidden_dim+1)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim+1)/2), 1)
                                                )

            else:
                self.encoder = nn.Sequential(nn.Linear(hidden_dim+1, int((hidden_dim+1)/2)),
                                             #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                             nn.ELU(),
                                             nn.Linear(int((hidden_dim+1)/2), 4)
                                             )

    def forward(self, X, Hct=None, first=False, epoch=0):
        if self.hp.network.nn == 'linear':
            output = self.linear(X)
            params = self.encoder(torch.cat((output, Hct.unsqueeze(1)), axis=1))

        elif self.hp.network.nn in ['lstm', 'gru']:
            if self.hp.network.nn == 'lstm':
                output, (hn, cn) = self.rnn(X.unsqueeze(dim=2))
            else:
                output, hn = self.rnn(X.unsqueeze(dim=2))

            if self.hp.network.attention:
                score_ke = self.score_ke(output)
                score_ve = self.score_ve(output)
                score_vp = self.score_vp(output)
                score_dt = self.score_dt(output)

                '''
                if first:
                    plt.close()
                    plt.bar(np.arange(self.hp.max_rep), score_ke.squeeze().cpu().detach())
                    plt.title('scores {} epoch {}'.format('$k_{ep}$', epoch))
                    plt.savefig('param_ke2_epoch_{}'.format(epoch))
                    plt.close()
                    plt.bar(np.arange(self.hp.max_rep), score_ve.squeeze().cpu().detach())
                    plt.title('scores {} epoch {}'.format('$v_e$', epoch))
                    plt.savefig('param_ve2_epoch_{}'.format(epoch))
                    plt.close()
                    plt.bar(np.arange(self.hp.max_rep), score_vp.squeeze().cpu().detach())
                    plt.title('scores {} epoch {}'.format('$v_p$', epoch))
                    plt.savefig('param_vp2_epoch_{}'.format(epoch))
                    plt.close()
                    plt.bar(np.arange(self.hp.max_rep), score_dt.squeeze().cpu().detach())
                    plt.title('scores {} epoch {}'.format('$dt$', epoch))
                    plt.savefig('param_dt2_epoch_{}'.format(epoch))
                    plt.close()
                '''

                hidden_ke = torch.sum(output*score_ke, dim=1)
                hidden_ve = torch.sum(output*score_ve, dim=1)
                hidden_vp = torch.sum(output*score_vp, dim=1)
                hidden_dt = torch.sum(output*score_dt, dim=1)

                ke = self.encoder_ke(torch.cat((hidden_ke, Hct.unsqueeze(1)), axis=1)).squeeze()
                ve = self.encoder_ve(torch.cat((hidden_ve, Hct.unsqueeze(1)), axis=1)).squeeze()
                vp = self.encoder_vp(torch.cat((hidden_vp, Hct.unsqueeze(1)), axis=1)).squeeze()
                dt = self.encoder_dt(torch.cat((hidden_dt, Hct.unsqueeze(1)), axis=1)).squeeze()

            else:
                params = self.encoder(torch.cat((hn[-1], Hct.unsqueeze(1)), axis=1))

        ke_diff = self.hp.simulations.bounds[1, 0] - self.hp.simulations.bounds[0, 0]
        ve_diff = self.hp.simulations.bounds[1, 1] - self.hp.simulations.bounds[0, 1]
        vp_diff = self.hp.simulations.bounds[1, 2] - self.hp.simulations.bounds[0, 2]
        dt_diff = self.hp.simulations.bounds[1, 3] - self.hp.simulations.bounds[0, 3]

        if self.hp.network.attention:
            ke = self.hp.simulations.bounds[0, 0] + torch.sigmoid(ke.unsqueeze(1)) * ke_diff
            ve = self.hp.simulations.bounds[0, 1] + torch.sigmoid(ve.unsqueeze(1)) * ve_diff
            vp = self.hp.simulations.bounds[0, 2] + torch.sigmoid(vp.unsqueeze(1)) * vp_diff
            dt = self.hp.simulations.bounds[0, 3] + torch.sigmoid(dt.unsqueeze(1)) * dt_diff

        else:
            ke = self.hp.simulations.bounds[0, 0] + torch.sigmoid(params[:, 0].unsqueeze(1)) * ke_diff
            ve = self.hp.simulations.bounds[0, 1] + torch.sigmoid(params[:, 1].unsqueeze(1)) * ve_diff
            vp = self.hp.simulations.bounds[0, 2] + torch.sigmoid(params[:, 2].unsqueeze(1)) * vp_diff
            dt = self.hp.simulations.bounds[0, 3] + torch.sigmoid(params[:, 3].unsqueeze(1)) * dt_diff

        aif = torch.zeros(len(self.hp.aif.aif), X.size(0)).to(self.hp.device)
        aif[0] = self.hp.aif.aif['t0']
        aif[1] = self.hp.aif.aif['tr']
        aif[2] = self.hp.aif.aif['ab']/(1-Hct)
        aif[3] = self.hp.aif.aif['mb']
        aif[4] = self.hp.aif.aif['ae']
        aif[5] = self.hp.aif.aif['me']
        aif[6] = self.hp.aif.aif['ar']
        aif[7] = self.hp.aif.aif['mm']
        aif[8] = self.hp.aif.aif['mr']

        X_dw = functions.Cosine8AIF_ExtKety_deep_aif(self.hp.acquisition.timing, aif, ke, dt, ve, vp, self.hp.device)

        return X_dw, ke, dt, ve, vp


def load_optimizer(net, hp):
    if hp.training.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=hp.training.lr, weight_decay=hp.training.weight_decay)
    elif hp.training.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=hp.training.lr, momentum=0.9, weight_decay=hp.training.weight_decay)
    elif hp.training.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=hp.training.lr, weight_decay=hp.training.weight_decay)
    else:
        raise Exception(
            'No valid optimiser is chosen. Please select a valid optimiser: training.optim = ''adam'', ''sgd'', ''adagrad''')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=hp.training.lr_mult,
                                                     patience=hp.training.optim_patience, verbose=True)

    return optimizer, scheduler
