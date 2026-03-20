set -e

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory

pip install -e .
pip install -r requirements/metrics.txt -r requirements/deepspeed.txt
pip install transformers==4.57.6 # 5.x 版本保存的 tokenizer_config.json 在后续 verl 训练会报错

# 想办法装一下 flash-attn
pip install flash-attn==2.8.3 --no-build-isolation

cp -f ~/data/double7/training_data/llamafactory/* ./data/