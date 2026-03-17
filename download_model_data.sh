set -e
HF_TOKEN=

mkdir -p ~/LLM/double7
hf download double7/qwen2.5-3b.sft.mt.v1 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.sft.mt.v1 --token $HF_TOKEN
hf download double7/qwen2.5-3b.verl.genrm.v2 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.verl.genrm.v2 --token $HF_TOKEN
hf download double7/qwen2.5-3b.sft.mt_rm.v1 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.sft.mt_rm.v1 --token $HF_TOKEN

mkdir -p ~/data/double7
hf download double7/training_data --repo-type model --local-dir ~/data/double7/training_data --token $HF_TOKEN
