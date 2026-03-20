# openclaw
Your own personal AI assistant. Any OS. Any Platform. The lobster way. 🦞



# Getting Started

## Data Preparation

```
bash download_model_data.sh
```
## LlamaFactory Installation

```
bash prepare_llamafactory.sh
```

## run sft

```bash
# WANDB_PROJECT 没设置默认是 llamafactory
export WANDB_NAME=qwen2.5-7b-mt_rm-sft-v1

FORCE_TORCHRUN=1  llamafactory-cli train path/to/openclaw/llamafactory/mt_rm/qwen_full_sft.v1.yaml
# nohup 版本
# FORCE_TORCHRUN=1 nohup llamafactory-cli train examples/train_full/qwen_full_sft.yaml > sft_train.log 2>&1 &
```

```bash
# 训完后删除检查点
rm -rf /ckpt/double7/ckpt/Qwen/Qwen2.5-7B/sft/mt_rm/v1/checkpoint-*
```

## run verl

先启动 ray 集群

```bash
# 这个是单机8卡，如果跑 2 node 16卡，需要改一下 NNODES=2
bash verl/qwen7b/mt_rm/run_train.mt_rm.v1.sh
```

训完需要手动 merge 检查点
```
python -m verl.model_merger merge --backend fsdp --local_dir $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/unmerged/v1 --target_dir $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/v1

rm -r $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/unmerged/v1
```