#!/bin/bash
# Example: disaggregated / async GRPO on GCU with the ECCL checkpoint engine
#
# The colocated acceptance baseline (baseline_grpo_gsm8k_enflame.sh) keeps the
# FL-validated configuration untouched. This script is the reference for the
# separate actor / rollout layout, where weights move from the trainer to the
# rollout workers through the ECCL checkpoint engine that verl_hardware_plugin
# registers as "eccl" ("nccl" is a compatibility alias). It follows the layout
# of verl's tests/special_e2e/run_v1_separate_async.sh and uses only
# verl + torch_gcu + ECCL; it does not depend on the FL stack.
#
# Ray workers do not inherit shell exports, so every setting below is also
# passed through ray_init.runtime_env.env_vars.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES=1
export VERL_PLATFORM="${VERL_PLATFORM:-enflame}"
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}

########################### user-adjustable ###########################
NUM_GPUS=${NUM_GPUS:-8}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-$((NUM_GPUS / 2))}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-$((NUM_GPUS - N_GPUS_TRAINING))}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp2}

export TOPS_VISIBLE_DEVICES=${TOPS_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}

DATA_DIR=${DATA_DIR:-/workspace/gsm8k}
MODEL_DIR=${MODEL_DIR:-/workspace/Qwen3-0.6B}

max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
parameter_sync_step=${PARAMETER_SYNC_STEP:-4}
train_batch_size=$((parameter_sync_step * ppo_mini_batch_size))
total_training_steps=${TOTAL_TRAINING_STEPS:-20}
rollout_tp=${ROLLOUT_TP:-1}
rollout_n=${ROLLOUT_N:-5}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-1024}

if ((N_GPUS_TRAINING <= 0 || N_GPUS_ROLLOUT <= 0 || N_GPUS_TRAINING + N_GPUS_ROLLOUT != NUM_GPUS)); then
    echo "Invalid GCU split: total=${NUM_GPUS}, training=${N_GPUS_TRAINING}, rollout=${N_GPUS_ROLLOUT}"
    exit 1
fi
if ((N_GPUS_ROLLOUT % rollout_tp != 0)); then
    echo "N_GPUS_ROLLOUT must be divisible by ROLLOUT_TP"
    exit 1
fi

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0.6b_grpo_async_eccl_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

RAY_INIT=(
 +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=\'1\'
 +ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES=\'1\'
 +ray_kwargs.ray_init.runtime_env.env_vars.TOPS_VISIBLE_DEVICES="'${TOPS_VISIBLE_DEVICES}'"
 +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALL2ALL_BACKEND=allgather_reducescatter
 +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=\'verl_hardware_plugin\'
 +ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM=\'enflame\'
 +ray_kwargs.ray_init.runtime_env.env_vars.TORCHGCU_INDUCTOR_ENABLE=\'0\'
 +ray_kwargs.ray_init.runtime_env.env_vars.TORCHDYNAMO_DISABLE=\'1\'
 +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENABLE_V1_MULTIPROCESSING=\'0\'
 +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_ECCL_AVOID_RECORD_STREAMS=\'1\'
)

COMMON=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=separate_async
    trainer.v1.separate_async.num_warmup_batches=1
    trainer.v1.separate_async.parameter_sync_step=${parameter_sync_step}
    trainer.v1.sampler.max_off_policy_threshold=8
    trainer.v1.sampler.max_off_policy_strategy=drop
    transfer_queue.enable=True
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$DATA_DIR/train.parquet']"
    data.val_files="['$DATA_DIR/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.gen_batch_size=1
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=left
    data.return_raw_chat=True
    actor_rollout_ref.model.path="$MODEL_DIR"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.nnodes=1
    actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576
    # ECCL checkpoint engine: registered by verl_hardware_plugin for GCU.
    actor_rollout_ref.rollout.checkpoint_engine.backend=eccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${bucket_megabytes}
    reward.reward_manager.name=dapo
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.total_epochs=1
    trainer.total_training_steps=${total_training_steps}
)

if [[ "${ACTOR_STRATEGY}" == "fsdp2" ]]; then
    STRATEGY=(
        actor_rollout_ref.actor.strategy=fsdp2
        actor_rollout_ref.actor.fsdp_config.fsdp_size=2
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    )
elif [[ "${ACTOR_STRATEGY}" == "fsdp" ]]; then
    STRATEGY=(
        actor_rollout_ref.actor.strategy=fsdp
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    )
else
    echo "Unknown ACTOR_STRATEGY=${ACTOR_STRATEGY}; expected fsdp or fsdp2"
    exit 1
fi

echo "Running ECCL disaggregated/async GRPO: ${N_GPUS_TRAINING} training + ${N_GPUS_ROLLOUT} rollout GCUs"

python3 -m verl.trainer.main_ppo \
    "${RAY_INIT[@]}" \
    "${COMMON[@]}" \
    "${STRATEGY[@]}" \
    "$@" \
    2>&1 | tee "verl_demo_async.log"
