from __future__ import annotations

import pathlib

PATH_SRC = pathlib.Path(__file__).absolute().parent
PATH_PROJECT_ROOT = PATH_SRC.parent.absolute()
PATH_DATA = PATH_PROJECT_ROOT.joinpath("data")
DEEPSPEED_CONFIG_BASE = {
   #  "zero_optimization": {
   #      "stage": 3,
   #      "offload_optimizer": {
   #          "device": "cpu"
   #      },
   #      "offload_param": {
   #          "device": "cpu"
   #      },
   #      "allgather_partitions": True,
   #      "allgather_bucket_size": 2e8,
   #      "reduce_scatter": True,
   #      # "reduce_bucket_size": 2e8,
   #      "overlap_comm": True,
   #      "contiguous_gradients": True,
   #      "stage3_max_live_parameters": 1e9,
   #      "stage3_max_reuse_distance": 1e9,
   #      "stage3_prefetch_bucket_size": 1e7,
   #      "stage3_param_persistence_threshold": 1e5,
   #      "reduce_bucket_size": 1e7,
   #      "sub_group_size": 1e9
   # },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}