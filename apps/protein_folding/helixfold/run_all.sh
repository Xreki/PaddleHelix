#!/bin/bash

#SUFFIX=0417_94afa5

#pip uninstall -y paddlepaddle-gpu
#pip install /root/paddlejob/workspace/work/liuyiqun/AlphaFold/PaddleHelix/apps/protein_folding/paddle_whls/paddle_cuda116_${SUFFIX}/paddlepaddle_gpu-0.0.0.post116-cp38-cp38-linux_x86_64.whl

#log_dir="logs_cuda116_${SUFFIX}"
#log_dir="logs_cuda118_0510_ee1aa6_baseline"
#log_dir="logs_cuda118_0510_ee1aa6_amp_o2"
#log_dir="logs_cuda118_0512_d01c89_einsum"
#log_dir="logs_cuda118_0515_1019b2_cpp_einsum_layernorm"
#log_dir="logs_cuda118_0518_374797_flash_attn_off"
#log_dir="logs_cuda118_0519_645e81_flash_attn_o2_einsum_off"
#log_dir="logs_cuda117_0520_83a12b_layernorm_o2_cpp_einsum"
log_dir="logs_cuda117_0524_f2ed40_autotune_o2_fuse_linear"

mkdir -p ${log_dir}

#opt_version=002_AMP_O2
#opt_version=003_flash_attn_o2_einsum_off
#opt_version=004_layernorm_o2
#opt_version=004_layernorm_o2_cpp_einsum
opt_version=005_autotune_o2_fuse_linear

exp_name_array[0]=demo_initial_N1C1_dp1_dap1_bp1
exp_name_array[1]=demo_initial_N1C8_dp8_dap1_bp1
exp_name_array[2]=demo_initial_N1C2_dp1_dap1_bp2
exp_name_array[3]=demo_initial_N1C8_dp1_dap4_bp2
exp_name_array[4]=demo_finetune_N1C8_dp2_dap4_bp1_model5
exp_name_array[5]=demo_finetune_N1C8_dp8_dap1_bp1_model5
exp_name_array[6]=demo_finetune_N1C8_dp2_dap4_bp1_model1
exp_name_array[7]=demo_finetune_N1C8_dp8_dap1_bp1_model1
#exp_name_array[8]=demo_finetune_N1C1_dp1_dap1_bp1_model1

#export GLOG_vmodule=fused_gate_attention_op=4,fused_gate_attention=6
#export GLOG_vmodule=blaslt_impl=6,fused_gemm_epilogue_op=6
#export GLOG_v=6

precision="bf16"
amp_level="O1"

for i in {1..1}
do
  for name_id in {0..0}
  do
    exp_name=${exp_name_array[name_id]}
    log_filename=${opt_version}-${exp_name}-${precision}-${amp_level}-${i}.txt

    echo "============== TEST $i, ${log_dir}/${log_filename} =============="
    sh gpu_train.sh ${exp_name} ${precision} ${amp_level} #2>&1 | tee ${log_dir}/${log_filename}
    echo ""
  done
done
