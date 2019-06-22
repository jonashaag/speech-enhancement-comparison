<!--- 

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

-->

Deep Xi: A Deep Learning Approach to *A Priori* SNR Estimation. Used for Speech Enhancement and Robust ASR.
====

[Deep Xi](https://doi.org/10.1016/j.specom.2019.06.002) (where the Greek letter 'xi' or ξ is pronounced  /zaɪ/) is a deep learning approach to *a priori* SNR estimation that was proposed in [[1]](https://doi.org/10.1016/j.specom.2019.06.002) and is implemented in [TensorFlow](https://www.tensorflow.org/). Some of its use cases include:


* Minimum mean-square error (MMSE) approaches to **speech enhancement** like the MMSE short-time spectral amplitude (MMSE-STSA) estimator, the MMSE log-spectral amplitude (MMSE-LSA) estimator, and the Wiener filter (WF) approach.
* Estimate the ideal ratio mask **(IRM)** and the ideal binary mask **(IBM)**. 
* A **front-end for robust ASR**, as shown in **Figure 1**.

|![](./fig_front-end.png "Deep Xi as a front-end for robust ASR.")|
|----|
| <p align="center"> <b>Figure 1:</b> Deep Xi used as a front-end for robust ASR. The back-end (Deep Speech) is available <a href="https://github.com/mozilla/DeepSpeech">here</a>. The noisy speech magnitude spectrogram, as shown in <b>(a)</b>, is a mixture of clean speech with <i>voice babble</i> noise at an SNR level of -5 dB, and is the input to Deep Xi. Deep Xi estimates the <i>a priori</i> SNR, as shown in <b>(b)</b>. The <i>a priori</i> SNR estimate is used to compute an MMSE approach gain function, which is multiplied elementwise with the noisy speech magnitude spectrum to produce the clean speech magnitude spectrum estimate, as shown in <b>(c)</b>. <a href="https://github.com/anicolson/matlab_feat">MFCCs</a> are computed from the estimated clean speech magnitude spectrogram, producing the estimated clean speech cepstrogram, as shown in <b>(d)</b>. The back-end system, Deep Speech, computes the hypothesis transcript, from the estimated clean speech cepstrogram, as shown in <b>(e)</b>. </p> |


<!-- |![](./fig_reslstm.png "ResLSTM a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 3:</b> <a> ResLSTM </a> <i> a priori</i>  <a> SNR estimator.</a> </p> |

|![](./fig_resblstm.png "ResBLSTM a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 4:</b> <a> ResBLSTM </a> <i> a priori</i>  <a> SNR estimator.</a> </p> |
 -->

Current Models
-----
The ResLSTM and ResBLSTM networks used for Deep Xi in [1] have been replaced with a Residual Network that employs [causal dilated convolutional units](https://arxiv.org/pdf/1803.01271.pdf), otherwise known as a [temporal convolutional network (TCN)](https://arxiv.org/pdf/1803.01271.pdf). The Deep Xi network can be seen in **Figure 2**. It comprises of 2 million parameters.

|![](./fig_tcn.gif "Deep Xi a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 2:</b> <a> TCN </a> <i> a priori</i>  <a> SNR estimator.</a> </p> |

<!--
Trained models for **c2.7a** and **c1.13a** can be found in the *./model* directory. The trained model for **n1.9a** is to large to be stored on github. A model for **n1.9a** can be downloaded from [here](https://www.dropbox.com/s/wkhymfmx4qmqvg7/n1.5a.zip?dl=0). 
-->

Availability
-----

A trained network, version **3a**, can be found in the [*./model* directory](https://github.com/anicolson/DeepXi/tree/master/model/3a). 


Installation
-----

Prerequisites for GPU usage:

* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn)

To install:

1. `git clone https://github.com/anicolson/DeepXi.git`
2. `virtualenv --system-site-packages -p python3 ~/venv/DeepXi`
3. `source ~/venv/DeepXi/bin/activate`
4. `pip install --upgrade tensorflow-gpu`
5. `cd DeepXi`
6. `pip install -r requirements.txt`

If a GPU is not going to be used, replace step 4 with: `pip install --upgrade tensorflow`

How to Use the Deep Xi Scripts
-----
**Inference:**

```
python3 deepxi.py --infer 1 --out_type y --gain mmse-lsa --gpu 0
```
**y** for **--out_type** specifies enhanced speech .wav output. **mmse-lsa** specifies the used gain function (others include **mmse-stsa**, **wf**, **irm**, **ibm**, **srwf**, **cwf**).


**Training:**

```
python3 deepxi.py --train 1 --verbose 1 --gpu 0
```

**Retraining:**

```
python3 deepxi.py --train 1 --cont 1 --epoch 175 --verbose 1 --gpu 0
```

Other options can be found in the *deepxi.py* script.

Datasets Used For Training
-----
The .wav files used for training are single-channel, with a sampling frequency of 16 kHz.

The following speech datasets were used:
* The *train-clean-100* set from Librispeech corpus, which can be found [here](http://www.openslr.org/12/).
* The CSTR VCTK corpus, which can be found [here](https://datashare.is.ed.ac.uk/handle/10283/2651).
* The *si* and *sx* training sets from the TIMIT corpus, which can be found [here](https://catalog.ldc.upenn.edu/LDC93S1) (not open source).

The following noise datasets were used:
* The QUT-NOISE dataset, which can be found [here](https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/).
* The Nonspeech dataset, which can be found [here](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html). 
* The Environemental Background Noise dataset, which can be found [here](http://www.utdallas.edu/~nxk019000/VAD-dataset/).
* The noise set from the MUSAN corpus, which can be found [here](http://www.openslr.org/17/).
* Multiple packs from the FreeSound website, which can be found  [here](https://freesound.org/)

References
-----
Please cite the following when using Deep Xi:

[1] [A. Nicolson, K. K. Paliwal, Deep learning for minimum mean-square error approaches to speech enhancement, Speech Communication 111 (2019) 44 - 55, https://doi.org/10.1016/j.specom.2019.06.002.](https://doi.org/10.1016/j.specom.2019.06.002)
