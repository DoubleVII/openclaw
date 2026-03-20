set -e
HF_TOKEN=

mkdir -p ~/LLM/double7
hf download double7/qwen2.5-3b.sft.mt.v1 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.sft.mt.v1 --token $HF_TOKEN
hf download double7/qwen2.5-3b.verl.genrm.v2 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.verl.genrm.v2 --token $HF_TOKEN
hf download double7/qwen2.5-3b.sft.mt_rm.v1 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.sft.mt_rm.v1 --token $HF_TOKEN


# 上面是旧的，按道理再执行一次也会自动跳过

# 更新了sft数据
mkdir -p ~/data/double7
hf download double7/training_data --repo-type model --local-dir ~/data/double7/training_data --token $HF_TOKEN

mkdir -p ~/LLM/Qwen
hf download Qwen/Qwen2.5-7B --repo-type model --local-dir ~/LLM/Qwen/Qwen2.5-7B