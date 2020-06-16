import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utility import models
from sigproc import STFT, ISTFT, PadSignal
import time

eps = 1e-16

class IF(nn.Module):
    def __init__(self):
        super(IF, self).__init__()

    def forward(self, input):

        return input

# TCN-based speech enhancement
class TCNSENet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, window_size=32, shift_size=16,
                layer=8, stack=3, kernel=3, causal=False, non_causal_layer=0, noise_aware=False,
                nl_for_mask="sigmoid", absangle_input=False,dilated=True, stft_encdec=False,
                stft_encdec_trainable=False, delay_frame=False, online=False, init_dump=False):
        super(TCNSENet, self).__init__()

        # hyper parameters
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.causal = causal
        self.non_causal_layer = non_causal_layer
        self.noise_aware = noise_aware
        self.absangle_input = absangle_input
        self.stft_encdec = stft_encdec
        self.stft_encdec_trainable = stft_encdec_trainable
        self.delay_frame = delay_frame
        self.init_dump = init_dump

        self.pad_signal = PadSignal(self.window_size,self.shift_size)

        # input encoder
        if stft_encdec:
            self.encoder = STFT(self.enc_dim, self.window_size, self.shift_size, trainable=stft_encdec_trainable, online=online)
        else:
            self.encoder = nn.Conv1d(1, self.enc_dim, self.window_size, bias=False, stride=self.shift_size)
            if online:
                self.enc_buffer = EncBuffer(self.window_size, self.shift_size)

        # TCN separator
        if self.noise_aware:
            self.TCN = models.TCN(self.enc_dim, self.enc_dim*2, self.feature_dim, self.feature_dim*4,
                                  self.layer, self.stack, self.kernel, causal=self.causal, non_causal_layer=self.non_causal_layer, dilated=dilated, online=online, delay=self.delay_frame, init_dump=self.init_dump)
            if stft_encdec:
                self.decoder_n = ISTFT(self.enc_dim, self.window_size, self.shift_size, trainable=stft_encdec_trainable, online=online)
            else:
                self.decoder_n = nn.ConvTranspose1d(self.enc_dim, 1, self.window_size, bias=False, stride=self.shift_size)
                if online:
                    self.dec_buffer_n = DecBuffer(self.window_size, self.shift_size)

        else:
            self.TCN = models.TCN(self.enc_dim, self.enc_dim, self.feature_dim, self.feature_dim*4,
                                  self.layer, self.stack, self.kernel, causal=self.causal, non_causal_layer=self.non_causal_layer, dilated=dilated, online=online, delay=self.delay_frame, init_dump=self.init_dump)

        self.receptive_field = self.TCN.receptive_field

        if nl_for_mask == "sigmoid":
            self.nl_for_mask = nn.Sigmoid()
        elif nl_for_mask == "relu":
            self.nl_for_mask = nn.ReLU()
        elif nl_for_mask == "prelu":
            self.nl_for_mask = nn.PReLU()
        elif nl_for_mask == "none":
            self.nl_for_mask = IF()

        # output decoder
        if stft_encdec:
            self.decoder_s = ISTFT(self.enc_dim, self.window_size, self.shift_size, trainable=stft_encdec_trainable, online=online)
        else:
            self.decoder_s = nn.ConvTranspose1d(self.enc_dim, 1, self.window_size, bias=False, stride=self.shift_size)
            if online:
                self.dec_buffer_s = DecBuffer(self.window_size, self.shift_size)


    def forward_online(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(0)
            input = input.unsqueeze(1)

        bin_size = self.enc_dim//2
        batch_size = 1

        ahead_frame_num = self.delay_frame
        ahead_sample_num = ahead_frame_num * self.shift_size + self.window_size
        delay_sample_num = ahead_sample_num - self.shift_size

        sp = 0
        ttl_frame_num = input.shape[2]//self.shift_size
        rest_sample_num = input.shape[2] - ttl_frame_num*self.shift_size

        input_pad = th.tensor(np.zeros([1,1,self.shift_size-rest_sample_num + delay_sample_num]),dtype=th.float,device=th.device(input.device))
        input = th.cat([input,input_pad],2)
        output_s = th.tensor(np.zeros([1,1,input.shape[2]]),dtype=th.float,device=th.device(input.device))
        output_n = th.tensor(np.zeros([1,1,input.shape[2]]),dtype=th.float,device=th.device(input.device))

        loop_num = input.shape[2] // self.shift_size

        if self.stft_encdec:
            self.encoder.set_buffer_device(input.device)
            self.decoder_s.set_buffer_device(input.device)
            self.decoder_n.set_buffer_device(input.device)
        else:
            self.enc_buffer.set_buffer_device(input.device)
            self.dec_buffer_s.set_buffer_device(input.device)
            self.dec_buffer_n.set_buffer_device(input.device)

        self.TCN.set_buffer_device(input.device)

        self.enc_output_buffer = th.tensor(np.zeros([1,self.enc_dim,self.delay_frame+1]),dtype=th.float,device=th.device(input.device))

        elapsed_time_list = []
        for i in range(loop_num):
            start = time.time()

            self.enc_output_buffer[:,:,:-1] = self.enc_output_buffer[:,:,1:]

            if self.stft_encdec:
                enc_output = self.encoder.forward_online(input[:,:,sp:sp+self.shift_size])
            else:
                enc_input = self.enc_buffer.get_enc_input(input[:,:,sp:sp+self.shift_size])
                enc_output = self.encoder(enc_input)

            self.enc_output_buffer[:,:,-1:] = enc_output


            if self.absangle_input:
                enc_output_abs = th.sqrt(enc_output[:,:bin_size,:]**2+enc_output[:,bin_size:,:]**2 + eps)
                enc_output_angle = th.atan2(enc_output[:,bin_size:,:],enc_output[:,:bin_size,:])
                tcn_input = th.cat([enc_output_abs,enc_output_angle],1)
            else:
                tcn_input = enc_output


            enc_output = self.enc_output_buffer[:,:,:1]

            if self.noise_aware:
                masks = self.nl_for_mask(self.TCN.forward_online(tcn_input)).view(batch_size, 2, self.enc_dim, -1)
            else:
                masks = self.nl_for_mask(self.TCN.forward_online(tcn_input)).view(batch_size, 1, self.enc_dim, -1)

            masked_output = enc_output.unsqueeze(1) * masks


            if self.stft_encdec:
                output_s_tmp = self.decoder_s.forward_online(masked_output[:,0,:,:])
            else:
                dec_output_s = self.decoder_s(masked_output[:,0,:,:])
                output_s_tmp = self.dec_buffer_s.get_dec_output(dec_output_s)



            elapsed_time = time.time() - start
            print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            elapsed_time_list.append(elapsed_time)
            output_s[:,:,sp:sp+self.shift_size] = output_s_tmp

            if self.noise_aware:
                if self.stft_encdec:
                    output_n[:,:,sp:sp+self.shift_size] = self.decoder_n.forward_online(masked_output[:,1,:,:]) 
                else:
                    dec_output_n = self.decoder_n(masked_output[:,1,:,:])
                    output_n[:,:,sp:sp+self.shift_size] = self.dec_buffer_n.get_dec_output(dec_output_n)

                mask_list = [masks[:,0,:,:],masks[:,1,:,:]]
            else:
                mask_list = [masks[:,0,:,:]]

            sp = sp + self.shift_size

        if self.noise_aware:
            output_list = [output_s[:,:,delay_sample_num:-(self.shift_size-rest_sample_num)],output_n[:,:,delay_sample_num:-(self.shift_size-rest_sample_num)]]
        else:
            output_list = [output_s[:,:,delay_sample_num:-(self.shift_size-rest_sample_num)]]


        elapsed_time_dict=dict()
        elapsed_time_mean = np.mean(elapsed_time_list)
        elapsed_time_std = np.std(elapsed_time_list)
        elapsed_time_dict["mean"] = elapsed_time_mean
        elapsed_time_dict["std"] = elapsed_time_std


        return output_list, elapsed_time_dict


    def forward(self, input):

        if input.dim() == 1:
            input = th.unsqueeze(input, 0)

        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)

        enc_output = self.encoder(output)

        if self.delay_frame:
            pad = th.tensor(np.zeros([batch_size,self.enc_dim,self.delay_frame]),dtype=th.float,device=input.device).type(input.type())
            enc_output = th.cat([pad, enc_output, pad], 2)


        if self.absangle_input:
            bin_size = enc_output.shape[1]//2
            enc_output_abs = th.sqrt(enc_output[:,:bin_size,:]**2+enc_output[:,bin_size:,:]**2 + eps)
            enc_output_angle = th.atan2(enc_output[:,bin_size:,:],enc_output[:,:bin_size,:])
            tcn_input = th.cat([enc_output_abs,enc_output_angle],1)
        else:
            tcn_input = enc_output


        if self.noise_aware:
            masks = self.nl_for_mask(self.TCN(tcn_input)).view(batch_size, 2, self.enc_dim, -1)
        else:
            masks = self.nl_for_mask(self.TCN(tcn_input)).view(batch_size, 1, self.enc_dim, -1)

        masked_output = enc_output.unsqueeze(1) * masks

        if self.delay_frame:
            masked_output = masked_output[:,:,:,self.delay_frame:-self.delay_frame]

        output_s = self.decoder_s(masked_output[:,0,:,:])

        output_s = output_s[:,:,self.window_size-self.shift_size:-(rest+self.window_size-self.shift_size)].contiguous()
        output_s = output_s.view(batch_size, -1)

        if self.noise_aware:
            output_n = self.decoder_n(masked_output[:,1,:,:])
            output_n = output_n[:,:,self.window_size-self.shift_size:-(rest+self.window_size-self.shift_size)].contiguous()
            output_n = output_n.view(batch_size, -1)
            output_list = [output_s,output_n]

            mask_list = [masks[:,0,:,:],masks[:,1,:,:]]
        else:
            output_list = [output_s]
            mask_list = [masks[:,0,:,:]]


        return output_list, mask_list


class EncBuffer(nn.Module):
    def __init__(self, window_size, stride):
        super(EncBuffer, self).__init__()
        self.window_size = window_size
        self.stride = stride

        self.input_buffer = th.tensor(np.zeros([1,1,self.window_size]),dtype=th.float)

    def set_buffer_device(self, device):
        self.input_buffer = self.input_buffer.to(device)

        return

    def get_enc_input(self, input):
        self.input_buffer[:,:,self.window_size-self.stride:] = input
        output = self.input_buffer.clone()
        self.input_buffer[:,:,:self.window_size-self.stride] = self.input_buffer[:,:,self.stride:]

        return output

class DecBuffer(nn.Module):
    def __init__(self, window_size, stride):
        super(DecBuffer, self).__init__()
        self.window_size = window_size
        self.stride = stride

        self.output_buffer = th.tensor(np.zeros([1,1,self.window_size]),dtype=th.float,device=th.device('cpu'))

    def set_buffer_device(self, device):
        self.output_buffer = self.output_buffer.to(device)

        return

    def get_dec_output(self, input):
        self.output_buffer[:,:,:self.window_size-self.stride] = self.output_buffer[:,:,self.stride:]
        self.output_buffer[:,:,self.window_size-self.stride:] = 0.0
        self.output_buffer = self.output_buffer +  input
        output = self.output_buffer[:,:,:self.stride]

        return output
