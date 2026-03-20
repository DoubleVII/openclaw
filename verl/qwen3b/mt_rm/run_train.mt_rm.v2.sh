set -e

# data
TRAIN_FILES="[$HOME/data/double7/training_data/towerx_v2_all_mt-code-block-think.verl/train.parquet,$HOME/data/double7/training_data/tower_zhen_qwen_sampling_group_score_gemini_distill.ranking_score_prompt.verl/train.sft_rl.aug.small.parquet]"
VAL_FILES=$HOME/data/double7/training_data/towerx_v2_all_mt-code-block-think.verl/test_zh.parquet
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=256
MAX_PROMPT_LENGTH=1280
MAX_RESPONSE_LENGTH=4096

# Actor/Optimization
ACTOR_MODEL_PATH=$HOME/LLM/double7/qwen2.5-3b.sft.mt_rm.v1
ACTOR_OPT_LR=6e-6
ACTOR_LR_SCHEDULER_TYPE=constant
ACTOR_MIN_LR_RATIO=null
ACTOR_WARMUP_STEPS=-1
ACTOR_PPO_MINI_BSZ=128
ACTOR_PPO_MICRO_BSZ_PER_GPU=32
ACTOR_LOSS_MODE=gspo
ACTOR_LOSS_AGG_MODE=seq-mean-token-mean
ACTOR_CLIP_RATIO_LOW=0.0003
ACTOR_CLIP_RATIO_HIGH=0.0004
ACTOR_STRATEGY=fsdp2
ACTOR_MODEL_DTYPE=bf16
ACTOR_ULYSSES_SP_SIZE=1
ACTOR_SFT_COEF=0.2 # Coefficient for SFT loss, see Appendix C. (Stabilizing Group Relative Policy Optimization) in the paper.

# Rollout
ROLLOUT_NAME=vllm # we use sglang for policy rollout and vllm for reward model rollout
ROLLOUT_MODE=sync
ROLLOUT_LOG_PROB_MB_PER_GPU=128
ROLLOUT_TENSOR_MP_SIZE=1
ROLLOUT_GPU_MEM_UTIL=0.8
ROLLOUT_N=4
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.7
VAL_TOP_K=-1
VAL_DO_SAMPLE=true

# Algorithm
ADV_ESTIMATOR=grpo
KL_COEF=0.001
NORM_ADV_BY_STD_IN_GRPO=False
ACTOR_USE_KL_LOSS=False

# Reward Model
REWARD_RESPONSE_LENGTH=8192
REWARD_PROMPT_LENGTH=2048
REWARD_MAX_NUM_BATCHED_TOKENS=12000
REWARD_ROLLOUT_NAME=vllm # we use sglang for policy rollout and vllm for reward model rollout
REWARD_ROLLOUT_MODE=sync
REWARD_GPU_MEM_UTIL=0.8
REWARD_TENSOR_MP_SIZE=1
REWARD_TEMPERATURE=0
REWARD_TOP_P=1.0
REWARD_TOP_K=-1
REWARD_SCORE_SCALE=0.01
MT_REWARD_SCORE_SCALE=0.01
RANKING_REWARD_SCORE_SCALE=0.05

# Reward Control
REWARD_MANAGER=naive
REWARD_ENABLE=True
REWARD_STRATEGY=SelfReward
REWARD_FREE_CACHE_ENGINE=True
REWARD_CUSTOM_PROCESSOR_PATH=reward_utils/rm_lib.py
REWARD_CUSTOM_PROCESSOR_NAME=MultiTaskSelfRewardProcessor
CUSTOM_REWARD_FN_PATH=reward_utils/rm_lib.py
CUSTOM_REWARD_FN_NAME=score_reward_fn
REWARD_KEEP_GROUP=True # dispatch the rollout candidates by group, must be `True` for GRRM
RESPONSE_EXTRACTOR_TYPE=codeblock
OVERLONG_BUFFER_ENABLE=True
OVERLONG_BUFFER_LEN=2048
OVERLONG_BUFFER_PENALTY_FACTOR=0.04
DEFAULT_REWARD=-${OVERLONG_BUFFER_PENALTY_FACTOR} # default reward if we fail to get a valid reward
ENABLE_LANGUAGE_DETECTION=True

# Training/Logging
PROJECT_NAME=self_reward_verl
EXPERIMENT_NAME=self_reward_gspo.v2
N_GPUS_PER_NODE=8
VAL_BEFORE_TRAIN=True
NNODES=1
SAVE_FREQ=200
TEST_FREQ=50
RESUME_MODE=auto
TOTAL_EPOCHS=1
DEFAULT_LOCAL_DIR=$HOME/ckpt/double7/ckpt/Qwen/Qwen2.5-3B/verl/mt_rm/unmerged/v2
LOG_VAL_GENERATIONS=32


ray job submit \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m \
    verl.trainer.main_ppo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_OPT_LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${ACTOR_LR_SCHEDULER_TYPE} \
    actor_rollout_ref.actor.optim.min_lr_ratio=${ACTOR_MIN_LR_RATIO} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${ACTOR_WARMUP_STEPS} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${ACTOR_LOSS_MODE} \
    +actor_rollout_ref.actor.sft_coef=${ACTOR_SFT_COEF} \
    actor_rollout_ref.actor.loss_agg_mode=${ACTOR_LOSS_AGG_MODE} \
    actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.clip_ratio_low=${ACTOR_CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_ULYSSES_SP_SIZE} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MB_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TENSOR_MP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    critic.optim.lr=1e-5 \
    critic.model.path=null \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO} \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.enable=${REWARD_ENABLE} \
    reward_model.strategy=${REWARD_STRATEGY} \
    +reward_model.keep_group=${REWARD_KEEP_GROUP} \
    +reward_model.rollout.free_cache_engine=${REWARD_FREE_CACHE_ENGINE} \
    +reward_model.rollout.name=${REWARD_ROLLOUT_NAME} \
    +reward_model.rollout.mode=${REWARD_ROLLOUT_MODE} \
    +reward_model.rollout.gpu_memory_utilization=${REWARD_GPU_MEM_UTIL} \
    +reward_model.rollout.tensor_model_parallel_size=${REWARD_TENSOR_MP_SIZE} \
    +reward_model.rollout.max_num_batched_tokens=${REWARD_MAX_NUM_BATCHED_TOKENS} \
    +reward_model.rollout.temperature=${REWARD_TEMPERATURE} \
    +reward_model.rollout.top_p=${REWARD_TOP_P} \
    +reward_model.rollout.top_k=${REWARD_TOP_K} \
    +reward_model.rollout.response_length=${REWARD_RESPONSE_LENGTH} \
    +reward_model.prompt_length=${REWARD_PROMPT_LENGTH} \
    +reward_model.score_scale_factor=${REWARD_SCORE_SCALE} \
    +reward_model.mt_score_scale_factor=${MT_REWARD_SCORE_SCALE} \
    +reward_model.ranking_score_scale_factor=${RANKING_REWARD_SCORE_SCALE} \
    +reward_model.default_reward=${DEFAULT_REWARD} \
    reward_model.model.path=${REWARD_MODEL_PATH} \
    +reward_model.custom_processor.path=${REWARD_CUSTOM_PROCESSOR_PATH} \
    +reward_model.custom_processor.name=${REWARD_CUSTOM_PROCESSOR_NAME} \
    +reward_model.custom_processor.extractor_type=${RESPONSE_EXTRACTOR_TYPE} \
    +reward_model.custom_processor.overlong_buffer.enable=${OVERLONG_BUFFER_ENABLE} \
    +reward_model.custom_processor.overlong_buffer.max_resp_len=${MAX_RESPONSE_LENGTH} \
    +reward_model.custom_processor.overlong_buffer.len=${OVERLONG_BUFFER_LEN} \
    +reward_model.custom_processor.overlong_buffer.penalty_factor=${OVERLONG_BUFFER_PENALTY_FACTOR} \
    +reward_model.custom_processor.enable_language_detection=${ENABLE_LANGUAGE_DETECTION} \
    custom_reward_function.path=${CUSTOM_REWARD_FN_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FN_NAME} \
    trainer.logger=[console,wandb] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}