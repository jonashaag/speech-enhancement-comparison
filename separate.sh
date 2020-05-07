#!/usr/bin/env bash

### config ###
dataset_name="dns_challenge"
tag="conv_tasnet_snrloss"
data_list='
blind_test_set
'
# data_list='
# synthetic_no_reverb
# synthetic_with_reverb
# real_recordings
# '
gpuid=0
mode="online" # offline, online, or online_debug


### main ###
for data in ${data_list}; do

  if [ $mode = "offline" ]; then
    ./nnet/separate.py ./exp/dns_challenge/${tag} --input ./data/${dataset_name}/${data}/noisy.scp --dump-dir ./eval/${dataset_name}/output_data_${tag}/${data}  --gpu ${gpuid}

  elif [ $mode = "online" ]; then
    ./nnet/separate.py ./exp/dns_challenge/${tag}  --online 1 --input ./data/${dataset_name}/${data}/noisy.scp --dump-dir ./eval/${dataset_name}/output_data_${tag}_online/${data}  --gpu ${gpuid}

  elif [ $mode = "online_debug" ]; then
    ./nnet/separate.py ./exp/dns_challenge/${tag}  --online 1 --online_debug 1 --input ./data/${dataset_name}/${data}/noisy.scp --dump-dir ./eval/${dataset_name}/output_data_${tag}_online/${data}  --gpu ${gpuid}

  else
    echo "not supported"
  fi

done
