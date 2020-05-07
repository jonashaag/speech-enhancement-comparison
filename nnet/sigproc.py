import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class STFT(nn.Module):
    def __init__(self, fftsize, window_size, stride, win_type="default", trainable=False, online=False):
        super(STFT, self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride

        if win_type=="default": # sin window
            self.window_func = np.sqrt(np.hanning(self.window_size))
        elif win_type=="hanning":
            self.window_func = np.hanning(self.window_size)


        fcoef_r = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        fcoef_i = np.zeros((self.fftsize//2 + 1, 1, self.window_size))

        for w in range(self.fftsize//2+1):
            for t in range(self.window_size):
                fcoef_r[w, 0, t] = np.cos(2. * np.pi * w * t / self.fftsize)
                fcoef_i[w, 0, t] = -np.sin(2. * np.pi * w * t / self.fftsize)


        fcoef_r = fcoef_r * self.window_func
        fcoef_i = fcoef_i * self.window_func

        self.fcoef_r = th.tensor(fcoef_r, dtype=th.float)
        self.fcoef_i = th.tensor(fcoef_i, dtype=th.float)

        self.encoder_r = nn.Conv1d(1, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)
        self.encoder_i = nn.Conv1d(1, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)


        self.encoder_r.weight = th.nn.Parameter(self.fcoef_r)
        self.encoder_i.weight = th.nn.Parameter(self.fcoef_i)

        if trainable:
            self.encoder_r.weight.requires_grad = True
            self.encoder_i.weight.requires_grad = True
        else:
            self.encoder_r.weight.requires_grad = False
            self.encoder_i.weight.requires_grad = False


        # for online
        if online:
            #self.input_buffer = th.tensor(np.zeros([1,1,self.window_size]),dtype=th.float,device=th.device('cpu'))
            self.input_buffer = th.tensor(np.zeros([1,1,self.window_size]),dtype=th.float)

        #import pdb; pdb.set_trace()

    def set_buffer_device(self, device):
        self.input_buffer = self.input_buffer.to(device)

        #import pdb; pdb.set_trace()

        return


    def forward(self, input):


        stft_stride = self.stride
        stft_window_size = self.window_size
        stft_fftsize = self.fftsize

        spec_r = self.encoder_r(input)
        spec_i = self.encoder_i(input)

        x_spec_real = spec_r[:,1:,:] # remove DC
        x_spec_imag = spec_i[:,1:,:] # remove DC
        output = th.cat([x_spec_real,x_spec_imag],dim=1)

        return output

    def forward_online(self, input):
        self.input_buffer[:,:,self.window_size-self.stride:] = input
        spec_r = self.encoder_r(self.input_buffer)
        spec_i = self.encoder_i(self.input_buffer)
        output = th.cat([spec_r[:,1:,:],spec_i[:,1:,:]],dim=1)
        self.input_buffer[:,:,:self.window_size-self.stride] = self.input_buffer[:,:,self.stride:]

        return output

class ISTFT(nn.Module):
    def __init__(self, fftsize, window_size, stride, win_type="default", trainable=False, online=False):
        super(ISTFT, self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride

        gain_ifft = (2.0*self.stride) / self.window_size


        if win_type=="default":
            self.window_func = gain_ifft * np.sqrt(np.hanning(self.window_size) )
        elif win_type=="hanning":
            self.window_func = gain_ifft * np.hanning(self.window_size)


        coef_cos = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        coef_sin = np.zeros((self.fftsize//2 + 1, 1, self.window_size))

        for w in range(self.fftsize//2+1):
            alpha = 1.0 if w==0 or w==fftsize//2 else 2.0
            alpha /= fftsize
            #print alpha
            for t in range(self.window_size):
                coef_cos[w, 0, t] = alpha * np.cos(2. * np.pi * w * t / self.fftsize)
                coef_sin[w, 0, t] = alpha * np.sin(2. * np.pi * w * t / self.fftsize)

        self.coef_cos = th.tensor(coef_cos * self.window_func, dtype=th.float)
        self.coef_sin = th.tensor(coef_sin * self.window_func, dtype=th.float)

        self.decoder_re = nn.ConvTranspose1d(self.fftsize//2+1, 1, self.window_size, bias=False, stride=self.stride)
        self.decoder_im = nn.ConvTranspose1d(self.fftsize//2+1, 1, self.window_size, bias=False, stride=self.stride)

        self.decoder_re.weight = th.nn.Parameter(self.coef_cos)
        self.decoder_im.weight = th.nn.Parameter(self.coef_sin)

        if trainable:
            self.decoder_re.weight.requires_grad = True
            self.decoder_im.weight.requires_grad = True
        else:
            self.decoder_re.weight.requires_grad = False
            self.decoder_im.weight.requires_grad = False

        # for online
        if online:
            self.output_buffer = th.tensor(np.zeros([1,1,self.window_size]),dtype=th.float,device=th.device('cpu'))
            self.pad_dc = th.tensor(np.zeros([1,1,1]),dtype=th.float,device=th.device('cpu'))

    def set_buffer_device(self, device):
        self.output_buffer = self.output_buffer.to(device)
        self.pad_dc = self.pad_dc.to(device)

        return

    def forward(self, input):

        batch_size = input.shape[0]
        frame_size = input.shape[2]

        stft_stride = self.stride
        stft_window_size = self.window_size
        stft_fft_size = self.fftsize

        pad_real_dc = th.tensor(np.zeros([batch_size, 1, frame_size]),dtype=th.float,device=th.device(input.device))
        pad_imag_dc = th.tensor(np.zeros([batch_size, 1, frame_size]),dtype=th.float,device=th.device(input.device))


        real_part = th.cat([pad_real_dc,input[:,:self.fftsize//2,:]],dim=1)
        imag_part = th.cat([pad_imag_dc,input[:,self.fftsize//2:,:]],dim=1)

        time_cos = self.decoder_re(real_part)
        time_sin = self.decoder_im(imag_part)

        output = time_cos - time_sin


        return output

    def forward_online(self, input):

        real_part = th.cat([self.pad_dc,input[:,:self.fftsize//2,:]],dim=1)
        imag_part = th.cat([self.pad_dc,input[:,self.fftsize//2:,:]],dim=1)

        time_cos = self.decoder_re(real_part)
        time_sin = self.decoder_im(imag_part)


        self.output_buffer[:,:,:self.window_size-self.stride] = self.output_buffer[:,:,self.stride:]
        self.output_buffer[:,:,self.window_size-self.stride:] = 0.0
        self.output_buffer = self.output_buffer +  time_cos - time_sin
        output = self.output_buffer[:,:,:self.stride]

        return output

class PadSignal(nn.Module):
    def __init__(self,window_size,shift_size):
        super(PadSignal, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size


    def forward(self, input):

        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)

        rest = self.shift_size - (nsample - self.window_size) % self.shift_size
        if rest > 0:
            pad = th.tensor(np.zeros([batch_size,1,rest]),dtype=th.float,device=input.device).type(input.type())
            #import pdb; pdb.set_trace()
            input = th.cat([input, pad], 2)

        pad_aux = th.tensor(np.zeros([batch_size,1,self.window_size-self.shift_size]),dtype=th.float,device=input.device).type(input.type())

        input = th.cat([pad_aux, input, pad_aux], 2)

        return input, rest
