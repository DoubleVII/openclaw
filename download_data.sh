set -e
HF_TOKEN=

# 更新了sft数据以及mt test data
mkdir -p ~/data/double7
hf download double7/training_data --repo-type dataset --local-dir ~/data/double7/training_data --token $HF_TOKEN


# 这两个是公开数据
hf download double7/TowerBlocks-MT-Ranking --repo-type dataset --local-dir ~/data/double7/parquet_data/TowerBlocks-MT-Ranking
hf download double7/MT_Ranking_Metric_Test --repo-type dataset --local-dir ~/data/double7/parquet_data/MT_Ranking_Metric_Test
