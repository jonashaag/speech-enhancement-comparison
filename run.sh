#!/bin/bash -eu

model="$1"
outdir="$2"
shift
shift

workdir=`mktemp -d`
trap "rm -rf $workdir" EXIT

case "$model" in
  tcnse)
    ../venv/bin/python tcnse/nnet/separate.py tcnse/exp/dns_challenge/conv_tasnet_snrloss --input "$@" --dump-dir "$outdir"
    ;;

  convtasnet)
    ../venv/bin/python conv-tasnet/nnet/separate.py conv-tasnet/pretrain/bn --fs 16000 --input "$@" --dump-dir "$outdir"
    ;;

  deepxi)
    mkdir -p $workdir/in $workdir/out
    cp "$@" $workdir/in
    . ../venv/bin/activate
    cd DeepXi
    env TEST_X_PATH=$workdir/in OUT_PATH=$workdir/out ./run.sh GPU=-1 VER="resnet-1.0c" INFER=1 GAIN="mmse-lsa" 
    find $workdir/out | xargs cp $outdir
    find $workdir/out -type f -exec cp {} \; $outdir
    ;;

  dct)
    D=END-TO-END-SPEECH-ENHANCEMENT-BASED-ON-DISCRETE-COSINE-TRANSFORM
    mkdir -p $workdir/models $workdir/data/noisy_testset_wav
    for f in `ls $D/ckpt/`; do
      cp $D/ckpt/$f $workdir/models/`echo $f | sed s/200000//g`
    done
    cp "$@" $workdir/data/noisy_testset_wav
    env DCT_WORKDIR=$workdir $D/env/bin/python $D/infer.py
    find $workdir/data/denoised/ -type f -exec cp {} $outdir \;
    ;;

  deepfeaturelosses)
    for f in "$@"; do
      sox $f -r 16000 -b 32 -e float $workdir/$f
    done
    ../venv2/bin/python SpeechDenoisingWithDeepFeatureLosses/senet_infer.py "$outdir" `ls $workdir`
    ;;

  convtasnet-asteroid)
    # todo output dir
    ../venv/bin/python asteroid-infer.py mpariente/asteroid conv_tasnet 'mpariente/ConvTasNet_WHAM!_sepnoisy' 8000 "$@"
    ;;

  source_separation)
    # todo
    env PYTHONPATH=source_separation ../venv/bin/python source_separation/source_separation/synthesize.py separate \
      AUDIO_FILE OUT_PATH MODEL_NAME PRETRAINED_PATH --sample_rate=16000
    ;;
esac
