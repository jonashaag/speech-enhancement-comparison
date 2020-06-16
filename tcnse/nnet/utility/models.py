import numpy as np
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle

class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True, online=False, layer_count=None, count=None,
                cum_frame_num_init=None, init_dump=False, nc_layer=False, last_nc=False):
        super(cLN, self).__init__()

        self.eps = eps
        self.layer_count = layer_count
        self.count = count
        self.init_dump = init_dump
        self.nc_layer = nc_layer
        self.last_nc = last_nc

        if trainable:
            self.gain = nn.Parameter(th.ones(1, dimension, 1))
            self.bias = nn.Parameter(th.zeros(1, dimension, 1))
        else:
            self.gain = Variable(th.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(th.zeros(1, dimension, 1), requires_grad=False)

        self.cum_frame_num_init = cum_frame_num_init

        #for online implementation
        if online:
            self.dimension = dimension

            if self.layer_count == -1:
                self.set_init_for_online('./init_dump/cln_init.pickle')
            else:
                if self.nc_layer:

                    if self.last_nc:
                        if self.count == 0:
                            self.set_init_for_online('./init_dump/cln_%d_%d.pickle' %(self.layer_count,self.count))
                        elif self.count == 1:
                            self.cum_frame_num = 0
                            self.cum_sum = 0.0
                            self.cum_pow_sum = 0.0

                    else:
                         self.set_init_for_online('./init_dump/cln_%d_%d.pickle' %(self.layer_count,self.count))

                else:
                    self.cum_frame_num = 0
                    self.cum_sum = 0.0
                    self.cum_pow_sum = 0.0



    def set_init_for_online(self,fname):
        with open(fname, mode='rb') as fi:
            data = pickle.load(fi)
        self.cum_frame_num = data["cum_frame_num"]
        self.cum_sum = data["cum_sum"][0]
        self.cum_pow_sum = data["cum_pow_sum"][0]

        return

    def forward_online(self,input):

        self.cum_frame_num += 1
        step_sum  = input.sum()
        step_pow_sum = input.pow(2).sum()

        self.cum_sum += step_sum
        self.cum_pow_sum += step_pow_sum
        cum_mean = self.cum_sum/(self.cum_frame_num*self.dimension)

        cum_var = (self.cum_pow_sum - 2*cum_mean*self.cum_sum) / (self.cum_frame_num*self.dimension) + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)

        return x * self.gain.type(x.type()) + self.bias.type(x.type())

    def forward(self, input):

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = th.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = th.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel*(time_step+1), channel)

        entry_cnt = th.tensor(entry_cnt,dtype=th.float,device=input.device).type(input.type())

        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)



        if self.init_dump:
            if self.layer_count==-1:
                init_dict = dict()
                init_dict["cum_frame_num"] = self.cum_frame_num_init
                init_dict["cum_sum"] = cum_sum[:,self.cum_frame_num_init-1].detach().cpu().numpy()
                init_dict["cum_pow_sum"] = cum_pow_sum[:,self.cum_frame_num_init-1].detach().cpu().numpy()
                dump_fname = "./init_dump/cln_init.pickle"
                with open(dump_fname, mode='wb') as fo:
                     pickle.dump(init_dict, fo)

            if self.nc_layer:

                if self.last_nc:
                    if self.count == 0:
                        init_dict = dict()
                        init_dict["cum_frame_num"] = self.cum_frame_num_init
                        init_dict["cum_sum"] = cum_sum[:,self.cum_frame_num_init-1].detach().cpu().numpy()
                        init_dict["cum_pow_sum"] = cum_pow_sum[:,self.cum_frame_num_init-1].detach().cpu().numpy()
                        dump_fname = "./init_dump/cln_%d_%d.pickle" %(self.layer_count,self.count)
                        with open(dump_fname, mode='wb') as fo:
                             pickle.dump(init_dict, fo)

                else:
                    init_dict = dict()
                    init_dict["cum_frame_num"] = self.cum_frame_num_init
                    init_dict["cum_sum"] = cum_sum[:,self.cum_frame_num_init-1].detach().cpu().numpy()
                    init_dict["cum_pow_sum"] = cum_pow_sum[:,self.cum_frame_num_init-1].detach().cpu().numpy()
                    dump_fname = "./init_dump/cln_%d_%d.pickle" %(self.layer_count,self.count)
                    with open(dump_fname, mode='wb') as fo:
                         pickle.dump(init_dict, fo)

        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, attention=False,
                dilation=1, skip=True, causal_ln=False, causal_conv=False,
                online=False, init_dump=False, layer_count=None, out_delay=False,
                last_nc=False, skip_buffer_size=None, conv_buffer_ep=None):
        super(DepthConv1d, self).__init__()

        self.kernel = kernel
        self.dilation = dilation
        self.causal_ln = causal_ln
        self.causal_conv = causal_conv
        self.skip = skip
        self.attention = attention

        self.init_dump = init_dump
        self.layer_count = layer_count
        self.out_delay = out_delay
        self.last_nc = last_nc
        self.skip_buffer_size = skip_buffer_size
        self.conv_buffer_ep = conv_buffer_ep

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)

        if self.causal_conv:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding

        if online:
            self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
              groups=hidden_channel,
              padding=0)
        else:
            self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
              groups=hidden_channel,
              padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

        kernel_h = (kernel-1)*dilation//2
        if self.causal_ln:
            self.reg1 = cLN(hidden_channel, eps=1e-08, online=online, layer_count=layer_count, count=0, cum_frame_num_init=conv_buffer_ep, init_dump=init_dump, nc_layer=not(causal_conv), last_nc=last_nc)
            self.reg2 = cLN(hidden_channel, eps=1e-08, online=online, layer_count=layer_count, count=1, cum_frame_num_init=conv_buffer_ep-kernel_h, init_dump=init_dump, nc_layer=not(causal_conv), last_nc=last_nc)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.attention:
            self.attention_mask = nn.Sigmoid()

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

        # for online
        if online:
            self.input_buffer = th.tensor(np.zeros([1,input_channel,(kernel - 1) * dilation+1]),dtype=th.float)
            self.conv_buffer = th.tensor(np.zeros([1,hidden_channel,(kernel - 1) * dilation+1]),dtype=th.float)


            if not causal_conv:
                if last_nc:
                    fname = "./init_dump/nclayer%d.pickle" %(layer_count)
                    with open(fname, mode='rb') as fi:
                        data = pickle.load(fi)

                    buf_sp = (self.input_buffer.shape[2]-1)//2
                    self.input_buffer[:,:,buf_sp:-1] = th.from_numpy(data['ibuf'].astype(np.float32)).clone()
                    self.conv_buffer[:,:,buf_sp:-1] = th.from_numpy(data['cbuf'].astype(np.float32)).clone()

                else:
                    self.skip_buffer = th.tensor(np.zeros([1,input_channel,skip_buffer_size]),dtype=th.float)
                    fname = "./init_dump/nclayer%d.pickle" %(layer_count)
                    with open(fname, mode='rb') as fi:
                        data = pickle.load(fi)
                    self.input_buffer[:,:,:-1] = th.from_numpy(data['ibuf'].astype(np.float32)).clone()
                    self.conv_buffer[:,:,:-1] = th.from_numpy(data['cbuf'].astype(np.float32)).clone()
                    self.skip_buffer[:,:,:-1] = th.from_numpy(data['sbuf'].astype(np.float32)).clone()


    def set_buffer_device(self, device):
        self.conv_buffer = self.conv_buffer.to(device)
        self.input_buffer = self.input_buffer.to(device)

        if self.out_delay:
            self.skip_buffer = self.skip_buffer.to(device)

        return

    def forward_online(self, input):

        self.input_buffer[:,:,-1:] = input

        if self.causal_ln:
            output = self.conv1d(input)
            output = self.nonlinearity1(output)
            output = self.reg1.forward_online(output)
            self.conv_buffer[:,:,-1:] = output
            output = self.dconv1d(self.conv_buffer)
            output = self.nonlinearity2(output)
            output = self.reg2.forward_online(output)
        else:

            print("not supported")
            import pdb; pdb.set_trace()


        if self.causal_conv:
            input_prev = input
        else:
            prev_idx = (self.kernel-1)*self.dilation//2
            input_prev = self.input_buffer[:,:,prev_idx:prev_idx+1].clone()


        self.conv_buffer[:,:,:-1] = self.conv_buffer[:,:,1:]
        self.input_buffer[:,:,:-1] = self.input_buffer[:,:,1:]

        residual = self.res_out(output)

        if self.attention:
            residual = input * self.attention_mask(residual)

        if self.skip:
            skip = self.skip_out(output)

            if self.out_delay:
                self.skip_buffer[:,:,-1:] = skip
                skip_out = self.skip_buffer[:,:,0:1].clone()
                self.skip_buffer[:,:,:-1] = self.skip_buffer[:,:,1:]
            else:
                skip_out = skip

            return residual, skip_out, input_prev
        else:
            return residual



    def forward(self, input):

        output = self.conv1d(input)
        output = self.nonlinearity1(output)
        dconv_input = self.reg1(output)

        if self.causal_conv:
            output = self.dconv1d(dconv_input)[:,:,:-self.padding]
            output = self.nonlinearity2(output)
            output = self.reg2(output)

        else:

            output = self.dconv1d(dconv_input)
            output = self.nonlinearity2(output)
            output = self.reg2(output)

        residual = self.res_out(output)

        if self.attention:
            residual = input * self.attention_mask(residual)


        if self.skip:
            skip = self.skip_out(output)

            if self.init_dump:
                kernel_h = (self.kernel-1)*self.dilation//2

                if not self.causal_conv:
                    if self.last_nc:
                        ep_f = self.conv_buffer_ep
                        sp_f = 0
                        dump_fname = "./init_dump/nclayer%d.pickle" %(self.layer_count)
                        init_dict = dict()
                        init_dict["ibuf"] = input[:,:,sp_f:ep_f].detach().cpu().numpy()
                        init_dict["cbuf"] = dconv_input[:,:,sp_f:ep_f].detach().cpu().numpy()
                        init_dict["sbuf"] = None
                        with open(dump_fname, mode='wb') as fo:
                             pickle.dump(init_dict, fo)

                    else:
                        ep_f = self.conv_buffer_ep
                        sp_f = ep_f - (kernel_h*2)
                        dump_fname = "./init_dump/nclayer%d.pickle" %(self.layer_count)
                        init_dict = dict()
                        init_dict["ibuf"] = input[:,:,sp_f:ep_f].detach().cpu().numpy()
                        init_dict["cbuf"] = dconv_input[:,:,sp_f:ep_f].detach().cpu().numpy()
                        init_dict["sbuf"] = skip[:,:,:ep_f-kernel_h].detach().cpu().numpy()
                        with open(dump_fname, mode='wb') as fo:
                             pickle.dump(init_dict, fo)


            return residual, skip
        else:
            return residual


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True,
                 causal=False, non_causal_layer=0, dilated=True, online=False, delay=None, init_dump=False):
        super(TCN, self).__init__()

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8, online=online, layer_count=-1, cum_frame_num_init=delay, init_dump=init_dump)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        count = 0
        skip_buffer_size = delay + 1
        conv_buffer_ep = delay

        for s in range(stack):
            for i in range(layer):
                if count < non_causal_layer:
                    causal_l = False
                    kernel_h = (kernel-1)*(2**i)//2
                    skip_buffer_size = skip_buffer_size - kernel_h
                else:
                    causal_l = causal

                if count < (non_causal_layer - 1):
                    skip_out_delay = True
                else:
                    skip_out_delay = False

                if count == (non_causal_layer-1):
                    last_nc = True
                else:
                    last_nc = False

                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal_ln=causal, causal_conv=causal_l, online=online, init_dump=init_dump,layer_count=count,out_delay=skip_out_delay,last_nc=last_nc,skip_buffer_size=skip_buffer_size,conv_buffer_ep=conv_buffer_ep))
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal_ln=causal, causal_conv=causal_l, online=online, init_dump=init_dump,layer_count=count,out_delay=skip_out_delay,last_nc=last_nc,skip_buffer_size=skip_buffer_size,conv_buffer_ep=conv_buffer_ep))

                if count < non_causal_layer:
                    conv_buffer_ep = conv_buffer_ep - kernel_h

                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
                count += 1

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )

        self.skip = skip

    def set_buffer_device(self, device):
        for i in range(len(self.TCN)):
            self.TCN[i].set_buffer_device(device)

        return


    def forward_online(self, input):

        output = self.BN(self.LN.forward_online(input))

        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip, input_prev = self.TCN[i].forward_online(output)
                output = input_prev + residual
                skip_connection = skip_connection + skip

        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i].forward_online(output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output

    def forward(self, input):

        # normalization
        output = self.BN(self.LN(input))

        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output
