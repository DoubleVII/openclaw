set -e
HF_TOKEN=

mkdir -p ~/LLM/double7
hf download double7/qwen2.5-3b.sft.mt.v1 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.sft.mt.v1 --token $HF_TOKEN
hf download double7/qwen2.5-3b.verl.genrm.v2 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.verl.genrm.v2 --token $HF_TOKEN
hf download double7/qwen2.5-3b.sft.mt_rm.v1 --repo-type model --local-dir ~/LLM/double7/qwen2.5-3b.sft.mt_rm.v1 --token $HF_TOKEN


mkdir -p ~/LLM/Qwen
hf download Qwen/Qwen2.5-7B --repo-type model --local-dir ~/LLM/Qwen/Qwen2.5-7B

mkdir -p ~/LLM/openai
hf download openai/gpt-oss-120b --repo-type model --local-dir ~/LLM/openai/gpt-oss-120b --ignore-pattern '["metal/","original/"]' --token $HF_TOKEN
