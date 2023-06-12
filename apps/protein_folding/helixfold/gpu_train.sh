#!/bin/bash

exp_name="$1"
precision=${2:-"bf16"}
amp_level=${3:-"O1"}

cd $(dirname $0)

root_path="$(pwd)"
PPFLEETX_PATH=/root/paddlejob/workspace/work/liuyiqun/PaddleFleetX

# export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
export PYTHONPATH=$root_path:${PPFLEETX_PATH}:$PYTHONPATH
# export PADDLE_NODE_NUM=$PADDLE_TRAINERS_NUM
# export PADDLE_NODE_NUM=1
TM_SCORE_BIN="$root_path/tools/tm_score"
LDDT_SCORE_BIN="$root_path/tools/lddt"
chmod +x $TM_SCORE_BIN
chmod +x $LDDT_SCORE_BIN

# Enable C++ enisum instead of python enisum
export FLAGS_new_einsum=0

# Enable fast layer_norm
export FLAGS_use_fast_math=1

# Enable/Disable bf16 optimization
export FLAGS_use_autotune=1

# Enable LayerNorm optimization
export FLAGS_use_fast_math=1

#if [ "${precision}" = "bf16" ]; then
#  export NVIDIA_TF32_OVERRIDE=0
#fi

train_af2_single() {
    start_step=1
    train_step=105
    CUDA_VISIBLE_DEVICES=0 python train.py \
            --tm_score_bin="$TM_SCORE_BIN" \
            --lddt_score_bin="$LDDT_SCORE_BIN" \
            --data_config=${data_config} \
            --train_config=${train_config} \
            --model_name=${model_name} \
            --init_model=${init_model} \
            --start_step=${start_step} \
            --train_step=${train_step} \
            --precision=${precision} \
            --amp_level=${amp_level} \
            --num_workers 6 \
            --seed 2022 \
            --batch_size=$batch_size \
            --dap_degree=$dap_degree \
            --bp_degree=$bp_degree \
            ${log_step} \
            ${eval_step} \
            ${save_step} \
            --model_dir="./debug_models" \
            --log_dir="./debug_log" \
            # &> ./debug_log/$exp_name.log
}


train_af2_distributed() {
    start_step=1
    train_step=105
    python -m paddle.distributed.launch train.py \
            --distributed \
            --tm_score_bin="$TM_SCORE_BIN" \
            --lddt_score_bin="$LDDT_SCORE_BIN" \
            --data_config=${data_config} \
            --train_config=${train_config} \
            --model_name=${model_name} \
            --init_model=${init_model} \
            --start_step=${start_step} \
            --train_step=${train_step} \
            --precision=${precision} \
            --amp_level=${amp_level} \
            --num_workers 6 \
            --seed 2022 \
            --batch_size=$batch_size \
            --dap_degree=$dap_degree \
            --bp_degree=$bp_degree \
            ${log_step} \
            ${eval_step} \
            ${save_step} \
            --model_dir="./debug_models" \
            --log_dir="./debug_log" \
            # &> ./debug_log/$exp_name.log
}


mkdir -p debug_log debug_models


### demo_initial_N1C1_dp1_dap1_bp1
{
    if [[ "$exp_name" == "demo_initial_N1C1_dp1_dap1_bp1" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=7

        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/initial.json"
        data_config="./data_configs/demo_valid.json"
        model_name="initial"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        train_af2_single
    fi
}


### demo_initial_N1C8_dp8_dap1_bp1
{
    if [[ "$exp_name" == "demo_initial_N1C8_dp8_dap1_bp1" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/initial.json"
        data_config="./data_configs/demo_valid.json"
        model_name="initial"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        train_af2_distributed
    fi
}


### demo_initial_N1C2_dp1_dap1_bp2
{
    if [[ "$exp_name" == "demo_initial_N1C2_dp1_dap1_bp2" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1

        batch_size=1
        dap_degree=1
        bp_degree=2
        train_config="./train_configs/initial.json"
        data_config="./data_configs/demo_valid.json"
        model_name="initial"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        train_af2_distributed
    fi
}


### demo_initial_N1C8_dp1_dap4_bp2
{
    if [[ "$exp_name" == "demo_initial_N1C8_dp1_dap4_bp2" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

        batch_size=1
        dap_degree=4
        bp_degree=2
        train_config="./train_configs/initial.json"
        data_config="./data_configs/demo_valid.json"
        model_name="initial"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        train_af2_distributed
    fi
}


### demo_finetune_N1C8_dp2_dap4_bp1_model5
{
    if [[ "$exp_name" == "demo_finetune_N1C8_dp2_dap4_bp1_model5" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        
        batch_size=1
        dap_degree=4
        bp_degree=1
        train_config="./train_configs/finetune.json"
        data_config="./data_configs/demo_valid.json"
        model_name="model_5"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        init_model="$root_path/data/params/params_model_5.npz"
        train_af2_distributed
    fi
}


### demo_finetune_N1C8_dp8_dap1_bp1_model5
{
    if [[ "$exp_name" == "demo_finetune_N1C8_dp8_dap1_bp1_model5" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        
        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/finetune.json"
        data_config="./data_configs/demo_valid.json"
        model_name="model_5"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        init_model="$root_path/data/params/params_model_5.npz"
        train_af2_distributed
    fi
}


### demo_finetune_N1C8_dp2_dap4_bp1_model1
{
    if [[ "$exp_name" == "demo_finetune_N1C8_dp2_dap4_bp1_model1" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        
        batch_size=1
        dap_degree=4
        bp_degree=1
        train_config="./train_configs/finetune.json"
        data_config="./data_configs/demo_valid.json"
        model_name="finetune"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        init_model="$root_path/data/params/params_model_1.npz"
        train_af2_distributed
    fi
}


### demo_finetune_N1C8_dp8_dap1_bp1_model1
{
    if [[ "$exp_name" == "demo_finetune_N1C8_dp8_dap1_bp1_model1" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        
        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/finetune.json"
        data_config="./data_configs/demo_valid.json"
        model_name="finetune"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        init_model="$root_path/data/params/params_model_1.npz"
        train_af2_distributed
    fi
}

### demo_finetune_N1C1_dp1_dap1_bp1_model1
{
    if [[ "$exp_name" == "demo_finetune_N1C1_dp1_dap1_bp1_model1" ]]; then
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        export PADDLE_NNODES=1
        # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0
        
        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/finetune.json"
        data_config="./data_configs/demo_valid.json"
        model_name="finetune"
        log_step="--log_step=20"
        eval_step="--eval_step=104"
        save_step="--save_step=1000"
        init_model="$root_path/data/params/params_model_1.npz"
        train_af2_distributed
    fi
}
