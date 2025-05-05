gpu_indices=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk '{print $1}')
gpu_indices=$(echo "$gpu_indices" | paste -sd "" - | sed 's/,$//')
export CUDA_VISIBLE_DEVICES=$gpu_indices

export WANDB_API_KEY=c3ad4cf16160791568f606c1d7325b0136ed3bfa


python run_ag.py