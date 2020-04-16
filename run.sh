#!/bin/bash

PROJ_DIR='deepxi'

case `hostname` in
"fist")  echo "Running on fist."
    SET_PATH='/mnt/ssd/deep_xi_training_set'
    DATA_PATH='/home/aaron/data/'$PROJ_DIR
    TEST_X_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noisy_speech'
    TEST_S_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_clean_speech'
    OUT_PATH='/home/aaron/out/'$PROJ_DIR
    MODEL_PATH='/home/aaron/model/'$PROJ_DIR
    ;;
"pinky-jnr")  echo "Running on pinky-jnr."
    SET_PATH='/home/aaron/set/deep_xi_training_set'
    DATA_PATH='/home/aaron/data/'$PROJ_DIR
    TEST_X_PATH='/home/aaron/set/deep_xi_test_set/test_noisy_speech'
    TEST_S_PATH='/home/aaron/set/deep_xi_test_set/test_clean_speech'
    OUT_PATH='/home/aaron/out/'$PROJ_DIR
    MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
    ;;
*) echo "This workstation is not known. Using default paths."
    SET_PATH='set'
    DATA_PATH='data'
    TEST_X_PATH='set/test_noisy_speech'
    OUT_PATH='out'
    MODEL_PATH='model'
   ;;
esac

get_free_gpu () {
    NUM_GPU=$( nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | wc -l )
    echo "$NUM_GPU total GPU/s."
    if [ $1 -eq 1  ]
    then
        echo 'Sleeping'
        sleep 1m
    fi
    while true
    do
        for (( gpu=0; gpu<$NUM_GPU; gpu++ ))
        do
            VAR1=$( nvidia-smi -i $gpu --query-gpu=pci.bus_id --format=csv,noheader )
            VAR2=$( nvidia-smi -i $gpu --query-compute-apps=gpu_bus_id --format=csv,noheader | head -n 1)
            if [ "$VAR1" != "$VAR2" ]
            then
                return $gpu
            fi
        done
        echo 'Waiting for free GPU.'
        sleep 1m
    done
}

NETWORK='TCN'
TRAIN=0
INFER=0
TEST=0

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            NETWORK)            NETWORK=${VALUE} ;;
            GPU)                GPU=${VALUE} ;;
            TRAIN)              TRAIN=${VALUE} ;;
            INFER)              INFER=${VALUE} ;;
            TEST)               TEST=${VALUE} ;;
            *)
    esac
done

WAIT=0
if [ -z $GPU ]
then
    get_free_gpu $WAIT
    GPU=$?
fi

if [ "$NETWORK" == 'ResLSTM' ]
then
    python3 main.py --ver               'reslstm-1a'    \
                    --network           'ResLSTM'       \
                    --d_model           512             \
                    --n_blocks          5               \
                    --max_epochs        100             \
                    --resume_epoch      0               \
                    --test_epoch        0               \
                    --mbatch_size       8               \
                    --sample_size       1000            \
                    --f_s               16000           \
                    --T_d               32              \
                    --T_s               16              \
                    --min_snr           -10             \
                    --max_snr           20              \
                    --out_type          'y'             \
                    --gain              'mmse-lsa'      \
                    --save_model        1               \
                    --log_iter          0               \
                    --eval_example      1               \
                    --train             $TRAIN          \
                    --infer             $INFER          \
                    --test              $TEST           \
                    --gpu               $GPU            \
                    --set_path          $SET_PATH       \
                    --data_path         $DATA_PATH      \
                    --test_x_path       $TEST_X_PATH    \
                    --test_s_path       $TEST_S_PATH    \
                    --out_path          $OUT_PATH       \
                    --model_path        $MODEL_PATH
fi

if [ "$NETWORK" == 'TCN' ]
then
    python3 main.py --ver               'tcn-1a'        \
                    --network           'TCN'           \
                    --d_model           256             \
                    --n_blocks          40              \
                    --d_f               64              \
                    --k                 3               \
                    --max_d_rate        16              \
                    --max_epochs        200             \
                    --resume_epoch      0               \
                    --test_epoch        20              \
                    --mbatch_size       8               \
                    --sample_size       1000            \
                    --f_s               16000           \
                    --T_d               32              \
                    --T_s               16              \
                    --min_snr           -10             \
                    --max_snr           20              \
                    --out_type          'y'             \
                    --gain              'mmse-lsa'      \
                    --save_model        1               \
                    --log_iter          0               \
                    --eval_example      1               \
                    --train             $TRAIN          \
                    --infer             $INFER          \
                    --test              $TEST           \
                    --gpu               $GPU            \
                    --set_path          $SET_PATH       \
                    --data_path         $DATA_PATH      \
                    --test_x_path       $TEST_X_PATH    \
                    --test_s_path       $TEST_S_PATH    \
                    --out_path          $OUT_PATH       \
                    --model_path        $MODEL_PATH
fi
