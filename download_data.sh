set -e
HF_TOKEN=

# 更新了sft数据
mkdir -p ~/data/double7
hf download double7/training_data --repo-type dataset --local-dir ~/data/double7/training_data --token $HF_TOKEN