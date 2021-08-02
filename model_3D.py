# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import functions

# np.random.seed(42)
# torch.manual_seed(42)


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_update = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                     out_channels=self.hidden_dim,  # for update_gate
                                     kernel_size=kernel_size,
                                     padding=self.padding,
                                     bias=self.bias)

        self.conv_reset = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=self.hidden_dim,  # for reset_gate
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        input_gates = torch.cat([input_tensor, h_cur], dim=1)

        update_gate = torch.sigmoid(self.conv_update(input_gates))
        reset_gate = torch.sigmoid(self.conv_reset(input_gates))

        input_cnm = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cnm = torch.tanh(self.conv_can(input_cnm))

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],  # (b,t,c,h,w)
                                              h_cur=h)

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DCE_NET(nn.Module):
    def __init__(self, hp):
        super(DCE_NET, self).__init__()
        self.hp = hp

        self.hp.acquisition.FAlistnet = torch.FloatTensor(np.expand_dims(self.hp.acquisition.FAlist, axis=1)).to(hp.device)

        if self.hp.network.nn == 'convlin':
            input_dim = len(hp.acquisition.timing)
            self.layers_local = nn.ModuleList()
            self.layers_global = nn.ModuleList()
            self.dilation = 2

            for feature_dim in self.hp.network.layers:
                self.layers_local.extend([nn.Conv2d(input_dim, feature_dim, 3, padding=1),
                                          nn.BatchNorm2d(feature_dim),
                                          nn.ELU()
                                          ])

                if self.hp.network.dual_path:
                    self.layers_global.extend([nn.Conv2d(input_dim, feature_dim, 3, padding=self.dilation, dilation=self.dilation),
                                               nn.BatchNorm2d(feature_dim),
                                               nn.ELU()
                                               ])

                    self.dilation *= 2

                input_dim = feature_dim

            self.conv_local = nn.Sequential(*self.layers_local)
            self.conv_global = nn.Sequential(*self.layers_global)

            if self.hp.network.dual_path:
                self.fc_layers = nn.Sequential(nn.Linear(2*self.hp.network.layers[-1], self.hp.network.layers[-1]),
                                               #nn.BatchNorm1d(160),
                                               nn.ELU()
                                               )

            self.encoder = nn.Sequential(nn.Linear(self.hp.network.layers[-1]+1, int((self.hp.network.layers[-1]+1)/2)),
                                         #nn.BatchNorm1d(int((self.hp.network.layers[-1]+1)/2)),
                                         nn.ELU(),
                                         nn.Linear(int((self.hp.network.layers[-1]+1)/2), 4)
                                         )

        elif self.hp.network.nn == 'convgru':
            self.feature_encoder = nn.Sequential(nn.Conv2d(160, 160, 3, padding=1),
                                                 nn.BatchNorm2d(160),
                                                 nn.ELU(),
                                                 nn.Conv2d(160, 80, 3, padding=1),
                                                 nn.BatchNorm2d(80),
                                                 nn.ELU())

            self.convgru = ConvGRU((160, 160), 1, self.hp.network.layers, (3, 3), len(self.hp.network.layers),
                                   torch.cuda.FloatTensor, batch_first=True, bias=True, return_all_layers=False)

            if self.hp.network.attention:
                self.score_ke = nn.Sequential(nn.Linear(self.hp.network.layers[-1], 1), nn.Softmax(dim=1))
                self.score_ve = nn.Sequential(nn.Linear(self.hp.network.layers[-1], 1), nn.Softmax(dim=1))
                self.score_vp = nn.Sequential(nn.Linear(self.hp.network.layers[-1], 1), nn.Softmax(dim=1))
                self.score_dt = nn.Sequential(nn.Linear(self.hp.network.layers[-1], 1), nn.Softmax(dim=1))

                self.encoder_ke = nn.Sequential(nn.Linear(self.hp.network.layers[-1]+1, int(self.hp.network.layers[-1]/2)),
                                                nn.ELU(),
                                                nn.Linear(int(self.hp.network.layers[-1]/2), 1)
                                                )
                self.encoder_ve = nn.Sequential(nn.Linear(self.hp.network.layers[-1]+1, int(self.hp.network.layers[-1]/2)),
                                                nn.ELU(),
                                                nn.Linear(int(self.hp.network.layers[-1]/2), 1)
                                                )
                self.encoder_vp = nn.Sequential(nn.Linear(self.hp.network.layers[-1]+1, int(self.hp.network.layers[-1]/2)),
                                                nn.ELU(),
                                                nn.Linear(int(self.hp.network.layers[-1]/2), 1)
                                                )
                self.encoder_dt = nn.Sequential(nn.Linear(self.hp.network.layers[-1]+1, int(self.hp.network.layers[-1]/2)),
                                                nn.ELU(),
                                                nn.Linear(int(self.hp.network.layers[-1]/2), 1)
                                                )

        elif self.hp.network.nn == 'unet':
            self.init_dim = self.hp.network.layers[0]

            self.encoder1 = nn.Sequential(nn.Conv2d(160, self.init_dim, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim, self.init_dim, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim),
                                          nn.ReLU()
                                          )

            self.encoder2 = nn.Sequential(nn.Conv2d(self.init_dim, self.init_dim*2, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*2),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim*2, self.init_dim*2, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*2),
                                          nn.ReLU()
                                          )

            self.encoder3 = nn.Sequential(nn.Conv2d(self.init_dim*2, self.init_dim*4, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*4),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim*4, self.init_dim*4, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*4),
                                          nn.ReLU()
                                          )

            self.encoder4 = nn.Sequential(nn.Conv2d(self.init_dim*4, self.init_dim*8, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*8),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim*8, self.init_dim*8, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*8),
                                          nn.ReLU()
                                          )

            self.encoder5 = nn.Sequential(nn.Conv2d(self.init_dim*8, self.init_dim*16, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*16),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim*16, self.init_dim*16, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*16),
                                          nn.ReLU()
                                          )

            self.decoder1 = nn.Sequential(nn.Conv2d(self.init_dim*2, self.init_dim, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim, self.init_dim, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim),
                                          nn.ReLU()
                                          )

            self.encoder = nn.Sequential(nn.Linear(self.hp.network.layers[0]+1, int((self.hp.network.layers[0]+1)/2)),
                                         #nn.BatchNorm1d(int(self.init_dim/2)),
                                         nn.ELU(),
                                         nn.Linear(int((self.hp.network.layers[0]+1)/2), 4),
                                         )

            self.decoder2 = nn.Sequential(nn.Conv2d(self.init_dim*4, self.init_dim*2, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*2),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim*2, self.init_dim*2, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*2),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(self.init_dim*2, self.init_dim, 2, stride=2)
                                          )

            self.decoder3 = nn.Sequential(nn.Conv2d(self.init_dim*8, self.init_dim*4, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*4),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim*4, self.init_dim*4, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*4),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(self.init_dim*4, self.init_dim*2, 2, stride=2)
                                          )

            self.decoder4 = nn.Sequential(nn.Conv2d(self.init_dim*16, self.init_dim*8, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*8),
                                          nn.ReLU(),
                                          nn.Conv2d(self.init_dim*8, self.init_dim*8, 3, padding=1),
                                          nn.BatchNorm2d(self.init_dim*8),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(self.init_dim*8, self.init_dim*4, 2, stride=2)
                                          )

            self.decoder5 = nn.ConvTranspose2d(self.init_dim*(2**self.hp.network.layers[1]),
                                               self.init_dim*(2**(self.hp.network.layers[1]-1)),
                                               kernel_size=2,
                                               stride=2
                                               )

    def forward(self, X, Hct=None):
        if self.hp.network.nn == 'convlin':
            output_local = self.conv_local(X)
            output = torch.moveaxis(output_local, 1, 3).reshape(-1, output_local.size(1))

            if self.hp.network.dual_path:
                output_global = self.conv_global(X)
                output_global = torch.moveaxis(output_global, 1, 3).reshape(-1, output_global.size(1))
                output = self.fc_layers(torch.cat((output, output_global), axis=1))

            params = self.encoder(torch.cat((output, Hct.reshape(-1, 1)), axis=1))

        elif self.hp.network.nn == 'convgru':
            X_encoded = self.feature_encoder(X)
            output, last_state_list = self.convgru(X_encoded.unsqueeze(2))

            if self.hp.network.attention:
                output = torch.moveaxis(output[0], 2, 4)
                score_ke = self.score_ke(output)
                score_ve = self.score_ve(output)
                score_vp = self.score_vp(output)
                score_dt = self.score_dt(output)
                hidden_ke = torch.sum(output*score_ke, dim=1)
                hidden_ve = torch.sum(output*score_ve, dim=1)
                hidden_vp = torch.sum(output*score_vp, dim=1)
                hidden_dt = torch.sum(output*score_dt, dim=1)
                ke = self.encoder_ke(torch.cat((hidden_ke, Hct.unsqueeze(3)), axis=3)).flatten()
                ve = self.encoder_ve(torch.cat((hidden_ve, Hct.unsqueeze(3)), axis=3)).flatten()
                vp = self.encoder_vp(torch.cat((hidden_vp, Hct.unsqueeze(3)), axis=3)).flatten()
                dt = self.encoder_dt(torch.cat((hidden_dt, Hct.unsqueeze(3)), axis=3)).flatten()
            else:
                output = torch.moveaxis(last_state_list[0][0], 1, 3)
                params = self.fc_layers(torch.cat((output, Hct.unsqueeze(3)), axis=3).view(-1, output.shape[-1]+1))

        elif self.hp.network.nn == 'unet':
            unet_dict = {}
            unet_dict['top'] = self.encoder1
            unet_dict['down1'] = self.encoder2
            unet_dict['down2'] = self.encoder3
            unet_dict['down3'] = self.encoder4
            unet_dict['down4'] = self.encoder5
            unet_dict['up1'] = self.decoder1
            unet_dict['up2'] = self.decoder2
            unet_dict['up3'] = self.decoder3
            unet_dict['up4'] = self.decoder4
            unet_dict['bottom'] = self.decoder5

            unet_dict['output0'] = unet_dict['top'](X)
            for i in range(self.hp.network.layers[1]):
                unet_dict['output'+str(i+1)] = unet_dict['down'+str(i+1)](F.max_pool2d(unet_dict['output'+str(i)], 2))

            unet_dict['output_up'+str(i+1)] = unet_dict['bottom'](unet_dict['output'+str(i+1)])

            for i in reversed(range(self.hp.network.layers[1])):
                unet_dict['output_up'+str(i)] = unet_dict['up'+str(i+1)](torch.cat((unet_dict['output'+str(i)], unet_dict['output_up'+str(i+1)]), axis=1))

            output = torch.moveaxis(unet_dict['output_up0'], 1, 3).reshape(-1, unet_dict['output_up0'].size(1))
            params = self.encoder(torch.cat((output, Hct.reshape(-1, 1)), axis=1))

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

        aif = torch.zeros(len(self.hp.aif.aif), X.size(0)*X.size(2)*X.size(3)).to(self.hp.device)
        aif[0] = self.hp.aif.aif['t0']
        aif[1] = self.hp.aif.aif['tr']
        aif[2] = self.hp.aif.aif['ab']/(1-Hct.reshape(-1))
        aif[3] = self.hp.aif.aif['mb']
        aif[4] = self.hp.aif.aif['ae']
        aif[5] = self.hp.aif.aif['me']
        aif[6] = self.hp.aif.aif['ar']
        aif[7] = self.hp.aif.aif['mm']
        aif[8] = self.hp.aif.aif['mr']

        Conc = functions.Cosine8AIF_ExtKety_deep_aif(self.hp.acquisition.timing, aif, ke, dt, ve, vp, self.hp.device).view(-1, X.size(2), X.size(3), X.size(1))
        Conc = torch.moveaxis(Conc, 3, 1)

        ke = ke.view(X.size(0), X.size(2), X.size(3))
        ve = ve.view(X.size(0), X.size(2), X.size(3))
        vp = vp.view(X.size(0), X.size(2), X.size(3))
        dt = dt.view(X.size(0), X.size(2), X.size(3))

        return Conc, ke, dt, ve, vp


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
