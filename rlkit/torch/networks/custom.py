import torch.nn as nn
import torch
import copy
from rlkit.torch.networks import CNN, Mlp
from rlkit.torch.core import PyTorchModule


class BaselineSeqCifar(PyTorchModule):
    def __init__(self, raw_vision, pred_status, memory_variation):
        super().__init__()
        self.raw_vision = raw_vision
        self.pred_status = pred_status
        self.memory_variation = memory_variation
        self.s_len = 3072 + 2 + self.pred_status
        self.cnn = CNN(input_width=32, input_height=32, input_channels=3, output_size=4,
                       kernel_sizes=[6, 4, 3],
                       n_channels=[16, 32, 32],
                       strides=[3, 2, 1],
                       paddings=[0, 0, 0],
                       output_activation=nn.LeakyReLU())
        self.mlp = Mlp(hidden_sizes=[64, ],
                       input_size=4 + 2,
                       output_size=3)
        if self.memory_variation == 0:
            self.mlp_tmp = Mlp(hidden_sizes=[64, ],
                               input_size=pred_status + 4 + 3,
                               output_size=pred_status)
        elif self.memory_variation == 1:
            self.mlp_ext = Mlp(hidden_sizes=[64, 64, ],
                               input_size=pred_status + 4 + 3,
                               output_size=pred_status,
                               output_activation=nn.Sigmoid())
        elif self.memory_variation == 2:
            self.mlp_h = Mlp(hidden_sizes=[64, ],
                             input_size=1 + 4 + 3,
                             output_size=1)
            self.mlp_s = Mlp(hidden_sizes=[64, ],
                             input_size=1 + 4 + 3,
                             output_size=1)

    def forward(self, input):
        b, l = input.size()
        if l > self.s_len:
            v = input[:, :3072]
            s = input[:, 3072:3074]
            pre_h = input[:, 3074:self.s_len][None, ...].contiguous()
            t = int((l - self.s_len) / 3075)
            pre_seq = input[:, self.s_len:].view(b, t, 3075)
            h_i = pre_h[0]
            for i in range(t):
                v_i = pre_seq[:, i, :3072]
                fv_i = self.cnn(v_i)
                a_i = pre_seq[:, i, 3072:3075]
                if self.memory_variation == 0:
                    fvah_i = torch.cat((h_i, fv_i, a_i), dim=1)
                    h_i = self.mlp_tmp(fvah_i)
                elif self.memory_variation == 1:
                    fvah_i = torch.cat((h_i, fv_i, a_i), dim=1)
                    h_i = self.mlp_ext(fvah_i)
                elif self.memory_variation == 2:
                    fvahung_i = torch.cat((h_i[:, 0:1], fv_i, a_i), dim=1)
                    fvasick_i = torch.cat((h_i[:, 1:2], fv_i, a_i), dim=1)
                    hung_i = self.mlp_h(fvahung_i)
                    sick_i = self.mlp_s(fvasick_i)
                    h_i = torch.cat((hung_i, sick_i), dim=1)

            h = h_i
            fv = self.cnn(v)
            if self.pred_status == 1:
                fvh_t = torch.cat((fv, h, s[:, 1:]), dim=1)
            elif self.pred_status == 2:
                fvh_t = torch.cat((fv, h), dim=1)
            q = self.mlp(fvh_t)
            return q, h
        else:
            v = input[:, :3072]
            s = input[:, 3072:3074]
            fv = self.cnn(v)
            fvs = torch.cat((fv, s), dim=1)
            q = self.mlp(fvs)
            return q, s[:, 0:self.pred_status]

    def get_fv(self, input):
        v = input[:, :3072]
        fv = self.cnn(v)
        return fv


class BaselineVCifar(PyTorchModule):
    def __init__(self, raw_vision, raw_status):
        super().__init__()
        self.raw_vision = raw_vision
        self.raw_status = raw_status
        if raw_status:
            num_node = 6
        else:
            num_node = 4
        self.cnn = CNN(input_width=32, input_height=32, input_channels=1, output_size=4,
                       kernel_sizes=[6, 4, 3],
                       n_channels=[16, 32, 32],
                       strides=[3, 2, 1],
                       paddings=[0, 0, 0],
                       output_activation=nn.LeakyReLU())
        self.mlp = Mlp(hidden_sizes=[64, ],
                       input_size=num_node,
                       output_size=3)

    def forward(self, input):
        if self.raw_status:
            v = input[:, :3072]
            s = input[:, 3072:]
            fv = self.cnn(v)
            fvs = torch.cat((fv, s), dim=1)
            q = self.mlp(fvs)
        else:
            v = input[:, :3072]
            fv = self.cnn(v)
            q = self.mlp(fv)
        return q

    def get_fv(self, input):
        v = input[:, :3072]
        fv = self.cnn(v)
        return fv


class BaselineSeq(PyTorchModule):
    def __init__(self, raw_vision, pred_status, memory_variation):
        super().__init__()
        self.raw_vision = raw_vision
        self.pred_status = pred_status
        self.memory_variation = memory_variation
        if raw_vision:
            self.s_len = 784 + 2 + self.pred_status
            self.cnn = CNN(input_width=28, input_height=28, input_channels=1, output_size=4,
                           kernel_sizes=[6, 4, 3],
                           n_channels=[16, 32, 32],
                           strides=[3, 2, 1],
                           paddings=[0, 0, 0],
                           output_activation=nn.LeakyReLU())
            self.mlp = Mlp(hidden_sizes=[64, ],
                           input_size=4 + 2,
                           output_size=3)
            if self.memory_variation == 0:
                self.mlp_tmp = Mlp(hidden_sizes=[64, ],
                                   input_size=pred_status + 4 + 3,
                                   output_size=pred_status)
            elif self.memory_variation == 1:
                self.mlp_ext = Mlp(hidden_sizes=[64, 64, ],
                                   input_size=pred_status + 4 + 3,
                                   output_size=pred_status,
                                   output_activation=nn.Sigmoid())
            elif self.memory_variation == 2:
                self.mlp_h = Mlp(hidden_sizes=[64, ],
                                 input_size=1 + 4 + 3,
                                 output_size=1)
                self.mlp_s = Mlp(hidden_sizes=[64, ],
                                 input_size=1 + 4 + 3,
                                 output_size=1)
        else:
            self.s_len = 4 + 2 + self.pred_status
            self.mlp = Mlp(hidden_sizes=[64, ],
                           input_size=4 + 2,
                           output_size=3)
            self.mlp_tmp = Mlp(hidden_sizes=[64, ],
                               input_size=pred_status + 4 + 3,
                               output_size=pred_status)

    def forward(self, input):
        if self.raw_vision:
            b, l = input.size()
            if l > self.s_len:
                v = input[:, :784]
                s = input[:, 784:786]
                pre_h = input[:, 786:self.s_len][None, ...].contiguous()
                t = int((l - self.s_len) / 787)
                pre_seq = input[:, self.s_len:].view(b, t, 787)
                h_i = pre_h[0]
                for i in range(t):
                    v_i = pre_seq[:, i, :784]
                    fv_i = self.cnn(v_i)
                    a_i = pre_seq[:, i, 784:787]
                    if self.memory_variation == 0:
                        fvah_i = torch.cat((h_i, fv_i, a_i), dim=1)
                        h_i = self.mlp_tmp(fvah_i)
                    elif self.memory_variation == 1:
                        fvah_i = torch.cat((h_i, fv_i, a_i), dim=1)
                        h_i = self.mlp_ext(fvah_i)
                    elif self.memory_variation == 2:
                        fvahung_i = torch.cat((h_i[:, 0:1], fv_i, a_i), dim=1)
                        fvasick_i = torch.cat((h_i[:, 1:2], fv_i, a_i), dim=1)
                        hung_i = self.mlp_h(fvahung_i)
                        sick_i = self.mlp_s(fvasick_i)
                        h_i = torch.cat((hung_i, sick_i), dim=1)

                h = h_i
                fv = self.cnn(v)
                if self.pred_status == 1:
                    fvh_t = torch.cat((fv, h, s[:, 1:]), dim=1)
                elif self.pred_status == 2:
                    fvh_t = torch.cat((fv, h), dim=1)
                q = self.mlp(fvh_t)
                return q, h
            else:
                v = input[:, :784]
                s = input[:, 784:786]
                fv = self.cnn(v)
                fvs = torch.cat((fv, s), dim=1)
                q = self.mlp(fvs)
                return q, s[:, 0:self.pred_status]
        else:
            b, l = input.size()
            if l > self.s_len:
                v = input[:, :4]
                s = input[:, 4:6]
                pre_h = input[:, 6:self.s_len][None, ...].contiguous()
                t = int((l - self.s_len) / 7)
                pre_seq = input[:, self.s_len:].view(b, t, 7)
                h_i = pre_h[0]
                for i in range(t):
                    va_i = pre_seq[:, i, :]
                    vah_i = torch.cat((h_i, va_i), dim=1)
                    h_i = self.mlp_tmp(vah_i)
                h = h_i
                if self.pred_status == 1:
                    vh_t = torch.cat((v, h, s[:, 1:]), dim=1)
                elif self.pred_status == 2:
                    vh_t = torch.cat((v, h), dim=1)
                q = self.mlp(vh_t)
                return q, h
            else:
                v = input[:, :4]
                s = input[:, 4:6]
                vh = torch.cat((v, s), dim=1)
                q = self.mlp(vh)
                return q, s[:, 0:self.pred_status]

    def get_fv(self, input):
        if self.raw_vision:
            v = input[:, :784]
            fv = self.cnn(v)
        else:
            fv = input[:, :4]
        return fv


class BaselineV(PyTorchModule):
    def __init__(self, raw_vision, raw_status):
        super().__init__()
        self.raw_vision = raw_vision
        self.raw_status = raw_status
        if raw_status:
            num_node = 6
        else:
            num_node = 4
        if raw_vision:
            self.cnn = CNN(input_width=28, input_height=28, input_channels=1, output_size=4,
                           kernel_sizes=[6, 4, 3],
                           n_channels=[16, 32, 32],
                           strides=[3, 2, 1],
                           paddings=[0, 0, 0],
                           output_activation=nn.LeakyReLU())
            self.mlp = Mlp(hidden_sizes=[64, ],
                           input_size=num_node,
                           output_size=3)
        else:
            self.mlp = Mlp(hidden_sizes=[64, ],
                           input_size=num_node,
                           output_size=3)

    def forward(self, input):
        if self.raw_vision:
            if self.raw_status:
                v = input[:, :784]
                s = input[:, 784:]
                fv = self.cnn(v)
                fvs = torch.cat((fv, s), dim=1)
                q = self.mlp(fvs)
            else:
                v = input[:, :784]
                fv = self.cnn(v)
                q = self.mlp(fv)
        else:
            if self.raw_status:
                vs = input
                q = self.mlp(vs)
            else:
                v = input[:, :4]
                q = self.mlp(v)
        return q

    def get_fv(self, input):
        if self.raw_vision:
            v = input[:, :784]
            fv = self.cnn(v)
        else:
            fv = input[:, :4]
        return fv


class BaselineV2(PyTorchModule):
    def __init__(self):
        super().__init__()
        self.cnn = CNN(input_width=28, input_height=28, input_channels=1, output_size=4,
                       kernel_sizes=[6, 4, 3],
                       n_channels=[16, 32, 32],
                       strides=[3, 2, 1],
                       paddings=[0, 0, 0],
                       output_activation=nn.LeakyReLU())
        self.mlp = Mlp(hidden_sizes=[64, ],
                       input_size=4,
                       output_size=3)

    def forward(self, input):
        v = input[:, :784]
        fv = self.cnn(v)
        q = self.mlp(fv)
        return q

    def get_fv(self, input):
        v = input[:, :784]
        fv = self.cnn(v)
        return fv


class BaselineV1(PyTorchModule):
    def __init__(self):
        super().__init__()
        self.cnn = CNN(input_width=28, input_height=28, input_channels=1, output_size=4,
                       kernel_sizes=[6, 4, 3],
                       n_channels=[16, 32, 32],
                       strides=[3, 2, 1],
                       paddings=[0, 0, 0],
                       output_activation=nn.LeakyReLU())
        self.mlp = Mlp(hidden_sizes=[64, ],
                       input_size=4+2,
                       output_size=3)

    def forward(self, input):
        v = input[:, :784]
        s = input[:, 784:]
        fv = self.cnn(v)
        fvs = torch.cat((fv, s), dim=1)
        q = self.mlp(fvs)
        return q

    def get_fv(self, input):
        v = input[:, :784]
        fv = self.cnn(v)
        return fv


class BaselineV0(PyTorchModule):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp(hidden_sizes=[64, ],
                       input_size=4+2,
                       output_size=3)

    def forward(self, input):
        vs = input
        q = self.mlp(vs)
        return q


class BaselineM0(PyTorchModule):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp(hidden_sizes=[64, ],
                       input_size=4 + 2,
                       output_size=3)
        '''  output, (h_n, c_n)
        output (batch, seq_len, hidden_size * num_directions)
        h_n (batch,num_layers * num_directions, hidden_size)
        c_n (batch, num_layers * num_directions, hidden_size)
        '''
        self.lstm = nn.LSTM(input_size=7, hidden_size=1, num_layers=1, batch_first=True)
        self.rnn = nn.RNN(input_size=7, hidden_size=1, num_layers=1, batch_first=True)
        self.mlp_tmp = Mlp(hidden_sizes=[64, ],
                           input_size=1 + 4 + 3,
                           output_size=1)

    def forward(self, input):
        b, l = input.size()
        ''' # this is to test sampling strategies
        v = input[:, :4]
        s = input[:, 4:6]
        h = input[:, 6:7]
        vh = torch.cat((v, s), dim=1)
        q = self.mlp(vh)
        return q, h
        '''
        if l > 4 + 2 + 1:
            ''' # option 2
            v = input[:, :4]
            s = input[:, 4:6]
            pre_h = input[:, 6:]
            h = self.mlp_tmp(pre_h)
            vh_t = torch.cat((v, h, s[:, 1:]), dim=1)
            q = self.mlp(vh_t)
            return q, s[:, 0:1]  # h
            '''
            ''' # option 3
            v = input[:, :4]
            s = input[:, 4:6]
            pre_h = input[:, 6:7][None, ...].contiguous()
            t = int((l - 7) / 7)
            pre_seq = input[:, 7:].view(b, t, 7)
            # pre_c = copy.deepcopy(pre_h)
            rnn_out = self.rnn(pre_seq, pre_h)
            h = rnn_out[0][:, -1, :]
            # vh_t = torch.cat((v, h, s[:, 1:]), dim=1)
            vh_t = torch.cat((v, pre_h[0], s[:, 1:]), dim=1)
            q = self.mlp(vh_t)
            return q, s[:, 0:1]  # h
            '''
            # option 3 rnn as mlp
            v = input[:, :4]
            s = input[:, 4:6]
            pre_h = input[:, 6:7][None, ...].contiguous()
            t = int((l - 7) / 7)
            pre_seq = input[:, 7:].view(b, t, 7)

            h_i = pre_h[0]
            for i in range(t):
                va_i = pre_seq[:, i, :]
                vah_i = torch.cat((h_i, va_i), dim=1)
                h_i = self.mlp_tmp(vah_i)

            h = h_i
            vh_t = torch.cat((v, h, s[:, 1:]), dim=1)
            q = self.mlp(vh_t)
            return q, h
        else:
            v = input[:, :4]
            s = input[:, 4:6]
            h = input[:, 6:]
            vh = torch.cat((v, h, s[:, 1:]), dim=1)
            q = self.mlp(vh)
            return q, h


