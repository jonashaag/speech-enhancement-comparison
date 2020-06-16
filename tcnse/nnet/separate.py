#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2018 Jian Wu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# wujian@2018
# modified by Yuichiro Koyama

import os
import argparse

import torch as th
import numpy as np

from network import TCNSENet

from libs.utils import load_json, get_logger
from libs.audio import WaveReader, write_wav, read_wav

logger = get_logger(__name__)


class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid, online=False):
        self.online = online
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")

        if online:
            # initialize online nnet
            nnet_id = self._load_nnet(cpt_dir, online=False, init_dump=True)
            self.nnet_id = nnet_id.to(self.device) if gpuid >= 0 else nnet_id
            self.nnet_id.eval()
            if not os.path.exists("./init_dump"):
                os.makedirs("./init_dump")
            zeros_id = th.tensor(np.zeros([16000,]), dtype=th.float32, device=self.device)
            sps, masks = self.nnet_id(zeros_id)

            if args.online_debug:
                nnet_offline = self._load_nnet(cpt_dir, online=False)
                self.nnet_offline = nnet_offline.to(self.device) if gpuid >= 0 else nnet_offline
                self.nnet_offline.eval()

        nnet = self._load_nnet(cpt_dir, online=online)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()


    def init_online(self, cpt_dir, gpuid):
        nnet = self._load_nnet(cpt_dir, online=True)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()


    def _load_nnet(self, cpt_dir, online=False, init_dump=False):
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = TCNSENet(**nnet_conf,online=online,init_dump=init_dump)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, samps):

        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            if self.online:
                sps, elapsed_time_dict = self.nnet.forward_online(raw)
                sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]

                if args.online_debug:
                    sps_offline, masks_offline = self.nnet_offline(raw)
                    sp_samps_offline = [np.squeeze(s.detach().cpu().numpy()) for s in sps_offline]

                    diff = sp_samps_offline[0] - sp_samps[0]
                    pow_offline = np.mean(sp_samps_offline[0]**2)
                    pow_diff = np.mean(diff**2)
                    SNR = 10.0*np.log10(pow_offline/pow_diff)
                    if SNR < 100:
                        print ("Difference is too high!")
                        import pdb; pdb.set_trace()

                    print("elapsed_time_mean : %.2f [ms]" %(elapsed_time_dict["mean"] * 1000) )
                    print("elapsed_time_std : %.2f [ms]" %(elapsed_time_dict["std"] * 1000) )
                    import pdb; pdb.set_trace()

            else:
                sps, masks = self.nnet(raw)
                sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
                sps, masks = self.nnet(raw)
            return sp_samps


def run(args):
    cpt_tag=os.path.basename(args.checkpoint)

    if args.online==1:
        computer = NnetComputer(args.checkpoint, args.gpu, online=True)
    else:
        computer = NnetComputer(args.checkpoint, args.gpu)

    mix_input = [(os.path.basename(f), (read_wav(f), 'foo')) for f in args.input]
    for key, mix_samps in mix_input:
        filepath_org = mix_samps[1]
        mix_samps = mix_samps[0]
        logger.info("Compute on utterance {}...".format(key))

        if args.online==1:
            # re-initialize
            computer.init_online(args.checkpoint, args.gpu)

        spks = computer.compute(mix_samps)

        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            if computer.nnet.causal:
                samps = samps * 1.00
                # samps = samps * 0.89 # consistently -1dB
                # samps = samps * 0.71 # consistently -3dB
                # samps = samps * 0.56 # consistently -3dB
                if np.max(np.abs(samps)) >= 1.0:

                    print("clipping!")

            else:
                if False:
                    samps = samps * norm / np.max(np.abs(samps))

            if idx == 0:
                fname = os.path.join(args.dump_dir, "{}.speech.wav".format(key))
                write_wav(fname,samps,fs=args.fs)

                if not computer.nnet.causal:
                    ip = np.dot(samps,mix_samps)
                    samps_scale_optimized = ip / np.dot(samps,samps) * samps
                    fname_scale_optimized = fname.replace(cpt_tag,"%s_scale_optimized" %(cpt_tag))
                    write_wav(fname_scale_optimized,samps_scale_optimized,fs=args.fs)

                if args.online == 1:
                    fname_submit = os.path.join("%s" %(args.dump_dir), os.path.basename(filepath_org))
                    fname_submit = fname_submit.replace("%s_online" %(cpt_tag),"%s_online/submit" %(cpt_tag))
                    write_wav(fname_submit,samps*0.89,fs=args.fs) # -1dB (to avoid clipping)

                    #import pdb; pdb.set_trace()

            elif idx == 1:
                fname = os.path.join(args.dump_dir, "{}.noise.wav".format(key))
                write_wav(fname,samps,fs=args.fs)

    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--input", type=str, required=True, help="Script for input waveform", nargs="+")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--fs", type=int, default=16000, help="Sample rate for mixture input")
    parser.add_argument(
        "--online", type=int, default=0, help="1: online 0: offline")
    parser.add_argument(
        "--online_debug", type=int, default=0, help="1: debug on 0: debug off")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="sps_tas",
        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)
