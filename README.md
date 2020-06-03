## TCN-based Speech Enhancement
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/us/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/3.0/us/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/us/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States License</a>.

This repository is based on [funcwj's repository](https://github.com/funcwj/conv-tasnet) and [naplab's repository](https://github.com/naplab/Conv-TasNet).



### Requirements
Tested in Python 3.7.
Please see [requirements.txt](requirements.txt) for other dependensies.


### Data preparation
Place the [DNS-Challenge folder](https://github.com/microsoft/DNS-Challenge) (or its symbolic link) into data folder.
e.g.,
```bash
$ cd data
$ ln -s [your path to DNS-Challenge] DNS-Challenge
```
Then execute make_filelist_testset.sh
```bash
$ ./make_filelist_testset.sh
```

### Inference

* configure separate.sh

* run separate.sh
```bash
$ ./separate.sh
```
### Paper
If you find this repository useful, please consider citing:

      @article{koyama2020exploring,
      title={Exploring the Best Loss Function for DNN-Based Low-latency Speech Enhancement with Temporal Convolutional Networks},
      author={Koyama, Yuichiro and Vuong, Tyler and Uhlich, Stefan and Raj, Bhiksha},
      journal={arXiv preprint arXiv:2005.11611},
      year={2020}
      }
