#!/bin/bash

dir=$PWD

data_list='
synthetic_no_reverb
synthetic_with_reverb
real_recordings
blind_test_set
'


for data in ${data_list}; do

  if [ $data = "synthetic_no_reverb" ]; then
    wavdir=${dir}/DNS-Challenge/datasets/test_set/synthetic/no_reverb

  elif [ $data = "synthetic_with_reverb" ]; then
    wavdir=${dir}/DNS-Challenge/datasets/test_set/synthetic/with_reverb

  elif [ $data = "real_recordings" ]; then
    wavdir=${dir}/DNS-Challenge/datasets/test_set/real_recordings

  elif [ $data = "blind_test_set" ]; then
    wavdir=${dir}/DNS-Challenge/datasets/blind_test_set

  else
    echo "not supported"
  fi

  if [ $data = "synthetic_no_reverb" ] || [ $data = "synthetic_with_reverb" ]; then

    find ${wavdir}/noisy -name '*.wav' | sort  > path.txt
    find ${wavdir}/noisy -name '*.wav' | sort  | \
    awk -F "/" '{ print $NF }' | awk -F ".wav" '{print $1}' > id.txt
    mkdir -p ./dns_challenge/${data}
    paste id.txt path.txt | sort > ./dns_challenge/${data}/noisy.scp

  else

    find ${wavdir}/ -name '*.wav' | sort  > path.txt
    find ${wavdir}/ -name '*.wav' | sort  | \
    awk -F "/" '{ print $NF }' | awk -F ".wav" '{print $1}' > id.txt
    mkdir -p ./dns_challenge/${data}
    paste id.txt path.txt | sort > ./dns_challenge/${data}/noisy.scp

  fi

done
rm path.txt
rm id.txt
