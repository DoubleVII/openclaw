# openclaw
Your own personal AI assistant. Any OS. Any Platform. The lobster way. 🦞



# Getting Started

## Data and Model Preparation

```
bash download_data.sh
bash download_model.sh
```
## LlamaFactory Installation

```
bash prepare_llamafactory.sh
```

## run sft

```bash
# 替换配置文件中的 $HOME 为真实路径
python path/to/openclaw/path_yaml.py path/to/openclaw/llamafactory/qwen7b/mt_rm/qwen7b_full_sft.v1.yaml

# WANDB_PROJECT 没设置默认是 llamafactory
export WANDB_NAME=qwen2.5-7b-mt_rm-sft-v1

FORCE_TORCHRUN=1  llamafactory-cli train path/to/openclaw/llamafactory/qwen7b/mt_rm/qwen7b_full_sft.v1.yaml
# nohup 版本
# FORCE_TORCHRUN=1 nohup llamafactory-cli train path/to/openclaw/llamafactory/qwen7b/mt_rm/qwen7b_full_sft.v1.yaml > sft_train.log 2>&1 &
```

```bash
# 训完后删除检查点
rm -rf $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/sft/mt_rm/v1/checkpoint-*
```

## run verl

先启动 ray 集群

```bash
# 这个是单机8卡，如果跑 2 node 16卡，需要改一下 NNODES=2
bash path/to/openclaw/verl/qwen7b/mt_rm/run_train.mt_rm.v1.sh
```

训完需要手动 merge 检查点
```
python -m verl.model_merger merge --backend fsdp --local_dir $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/unmerged/v1 --target_dir $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/v1

rm -r $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/unmerged/v1
```



# Evaluation

## Data and Model Preparation

需要下载 openai/gpt-oss-120b 模型，作为 evaluator

```bash
bash download_data.sh
bash download_model.sh
```

下载 BLEURT-20 模型
```
mkdir -p ~/bleurt
cd ~/bleurt
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
cd -
```

## install

```
git clone https://github.com/NJUNLP/GRRM.git
cd GRRM
pip install transformers==4.57.6 
pip install -e ".[infer,eval]" # or `uv sync --all-extras`

# 想办法装一下 flash-attn
pip install flash-attn==2.8.3 --no-build-isolation
```

复制数据，确保 `bash download_data.sh` 已正确执行
```
mkdir parquet_data
cp -r ~/data/double7/parquet_data/* parquet_data/
cp -r ~/data/double7/training_data/test_data_mt parquet_data/
```

**Ranking Accuracy Evaluation:**

```bash
# 单卡（80G显存）或双卡即可
# 需要比较久，建议 nohup
CUDA_VISIBLE_DEVICES=0,1 python -m eval.run_ranking_acc_eval \
    --data_id tower_zhen_ranking_testset,wmt_newstest2020_psqm,wmt_generalMT2022_enzh_mqm,wmt_generalMT2022_zhen_mqm,wmt_generalMT2022_ende_mqm,wmt_generalMT2022_enru_mqm,seedx_challenge_ranking \
    --model_path $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/v1 \
    --model_name Qwen2.5-7B-MT-GRRM.v1 \
    --temperature 0.3 \
    --top_p 0.9 \
    --max_new_tokens 8192 \
    --prompt_type ranking_score \
    --model_type grrm \
    --runs 4
```

**MT Evaluation:**

```bash
# 单卡（80G显存）或双卡即可
# 需要非常久，建议 nohup；
# 也可也拆开 data_id 运行:
# Group1: seedx_challenge_zhen,seedx_challenge_enzh,wmt23_de_en,wmt23_ja_en,wmt23_ru_en,wmt23_uk_en,wmt23_zh_en
# Group2: wmt24pp_en_de,wmt24pp_en_es,wmt24pp_en_fr,wmt24pp_en_it,wmt24pp_en_nl,wmt24pp_en_pt,wmt24pp_en_ja,wmt24pp_en_ko,wmt24pp_en_ru,wmt24pp_en_uk,wmt24pp_en_zh
CUDA_VISIBLE_DEVICES=0,1 python -m eval.run_mt_eval \
    --data_id seedx_challenge_zhen,seedx_challenge_enzh,wmt23_de_en,wmt23_ja_en,wmt23_ru_en,wmt23_uk_en,wmt23_zh_en,wmt24pp_en_de,wmt24pp_en_es,wmt24pp_en_fr,wmt24pp_en_it,wmt24pp_en_nl,wmt24pp_en_pt,wmt24pp_en_ja,wmt24pp_en_ko,wmt24pp_en_ru,wmt24pp_en_uk,wmt24pp_en_zh \
    --model_path $HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-7B/verl/mt_rm/v1 \
    --model_name Qwen2.5-7B-MT-GRRM.v1 \
    --temperature 0.3 \
    --top_p 0.9 \
    --max_new_tokens 8192 \
    --metrics '["bleurt","oss"]' \
    --prompt_type codeblock-think \
    --runs 4 \
    --bleurt_model_path ~/bleurt/BLEURT-20 \
    --oss_model_path ~/LLM/openai/gpt-oss-120b
```