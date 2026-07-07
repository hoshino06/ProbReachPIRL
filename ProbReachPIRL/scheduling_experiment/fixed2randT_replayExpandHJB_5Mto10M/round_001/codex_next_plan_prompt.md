You are controlling the next round of PIRL weight and collocation-distribution scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.
- For replay_expand experiments, start close to replay-HJB and expand the HJB collocation neighborhood cautiously.

Output:
- Write ONLY valid JSON to: /home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_002_plan.json
- Return exactly 2 candidate(s).

Schema:
{
  "round_note": "brief rationale",
  "candidates": [
    {
      "name": "short_unique_name",
      "start_checkpoint": "path/to/ckpt-N",
      "schedule_initial": [1.0, hjb0, bdr0],
      "schedule_final": [1.0, hjb1, bdr1],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-5,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.0,
      "pinn_expand_jitter_final": 0.05,
      "pinn_expand_center": 500000,
      "pinn_expand_sharpness": 1e-5,
      "pinn_expand_time_base": "local"
    }
  ]
}

Selection rules:
- Continue from the best safe checkpoint when reward and MC are stable.
- If reward or meanMC degraded, reduce weights or slow the schedule before trying larger weights.
- Increase HJB/BDR gradually.
- Increase replay_expand jitter gradually; prefer holding or backing off jitter before increasing HJB/BDR when reward or meanMC weakens.
- Candidate-level pinn_* fields are optional; omitted fields inherit [training_env] defaults from the TOML config.
- Use at most one TD3-restart control per round, unless all scheduling checkpoints collapsed.
- Do not repeat an existing start_checkpoint + schedule_initial + schedule_final + pinn_expand_jitter_final combination unless round_note explains why.

Advisor context from TOML:

Manual notes:
- This is the direct 5M-to-10M replay_expand comparison against fixed2randT_replayHJB_7Mto10M_restart2.
- The earlier fixed2randT_replayHJB_5Mto10M run collapsed when HJB/BDR were too large. The restart path improved only after using very small HJB/BDR values around 0.0001 to 0.002.
- Use the same fixedT TD3 5M baseline and randT reset as the replay-HJB restart experiments so the comparison isolates HJB collocation distribution.
- Start almost indistinguishable from replay-HJB. Use tiny jitter finals such as 0.005 or 0.01 before considering 0.02.
- If reward or meanMC degrades, prefer reducing HJB/BDR or holding jitter over expanding further.
- At 10M, compare primarily against restart2 hold0015_R3 and ramp00175_R3, not against collapsed low-MC branches with low mean|MC-V|.

Reference paths:
- logs/drift/README.md
- scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart.toml
- scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/all_results.json
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2.toml
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/all_results.json
- scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/all_results.json


Completed results JSON:
[
  {
    "name": "ramp0to0001_expand0005",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_000/ramp0to0001_expand0005/train/0704_1419_ramp0to0001_expand0005_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "ramp0to0001_expand0005",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.0001,
        0.0001
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.0,
      "pinn_expand_jitter_final": 0.005,
      "pinn_expand_center": 500000,
      "pinn_expand_sharpness": 1e-05,
      "pinn_expand_time_base": "local"
    },
    "effective_training_env": {
      "CASE": "drift",
      "DRIFT_DT": "0.1",
      "DRIFT_RESET_MIXTURE_PROBS": "0.3,0.3,0.4",
      "DRIFT_RESET_MODE": "mixture",
      "DRIFT_RESET_SCALE": "1.0",
      "DRIFT_RESET_T_MAX": "5.0",
      "DRIFT_RESET_T_MIN": "0.0",
      "DRIFT_RESET_T_MODE": "random",
      "HJB_LAPLACIAN_MODE": "loop",
      "INITIAL_EXPLORATION_POLICY": "policy",
      "METHOD": "scheduling",
      "NUM_WORKERS": "2",
      "PINN_EXPAND_CENTER": "500000",
      "PINN_EXPAND_JITTER_FINAL": "0.005",
      "PINN_EXPAND_JITTER_INITIAL": "0.0",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "1.0",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1399,
        "max_abs_mc_v": 0.9988,
        "mean_mc": 0.4116,
        "mean_v": 0.2718
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1391,
        "max_abs_mc_v": 0.88,
        "mean_mc": 0.5998,
        "mean_v": 0.4639
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10000000149011612,
      "Loss/RL": 0.001964109018445015,
      "Loss/HJB": 1.2686518430709839,
      "Loss/HJB_replay": 2.5483222007751465,
      "Loss/BDR": 0.25072962045669556,
      "Weights/RL": 1.0,
      "Weights/HJB": 9.999999747378752e-05,
      "Weights/BDR": 9.999999747378752e-05,
      "PINN/Replay Jitter": 0.004999999888241291
    }
  },
  {
    "name": "ramp0to0001_expand001",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_000/ramp0to0001_expand001/train/0704_1419_ramp0to0001_expand001_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "ramp0to0001_expand001",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.0001,
        0.0001
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.0,
      "pinn_expand_jitter_final": 0.01,
      "pinn_expand_center": 500000,
      "pinn_expand_sharpness": 1e-05,
      "pinn_expand_time_base": "local"
    },
    "effective_training_env": {
      "CASE": "drift",
      "DRIFT_DT": "0.1",
      "DRIFT_RESET_MIXTURE_PROBS": "0.3,0.3,0.4",
      "DRIFT_RESET_MODE": "mixture",
      "DRIFT_RESET_SCALE": "1.0",
      "DRIFT_RESET_T_MAX": "5.0",
      "DRIFT_RESET_T_MIN": "0.0",
      "DRIFT_RESET_T_MODE": "random",
      "HJB_LAPLACIAN_MODE": "loop",
      "INITIAL_EXPLORATION_POLICY": "policy",
      "METHOD": "scheduling",
      "NUM_WORKERS": "2",
      "PINN_EXPAND_CENTER": "500000",
      "PINN_EXPAND_JITTER_FINAL": "0.01",
      "PINN_EXPAND_JITTER_INITIAL": "0.0",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "1.0",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1351,
        "max_abs_mc_v": 0.9987,
        "mean_mc": 0.4108,
        "mean_v": 0.2758
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1353,
        "max_abs_mc_v": 0.8995,
        "mean_mc": 0.6001,
        "mean_v": 0.4686
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.23000000417232513,
      "Loss/RL": 0.005527892615646124,
      "Loss/HJB": 0.7536641955375671,
      "Loss/HJB_replay": 1.3131181001663208,
      "Loss/BDR": 0.36113086342811584,
      "Weights/RL": 1.0,
      "Weights/HJB": 9.999999747378752e-05,
      "Weights/BDR": 9.999999747378752e-05,
      "PINN/Replay Jitter": 0.009999999776482582
    }
  },
  {
    "name": "hold0001_expand001_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_000/ramp0to0001_expand001/train/0704_1419_ramp0to0001_expand001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_001/hold0001_expand001_R1/train/0705_0100_hold0001_expand001_R1_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "hold0001_expand001_R1",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_000/ramp0to0001_expand001/train/0704_1419_ramp0to0001_expand001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.0001,
        0.0001
      ],
      "schedule_final": [
        1.0,
        0.0001,
        0.0001
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.01,
      "pinn_expand_jitter_final": 0.01,
      "pinn_expand_center": 500000,
      "pinn_expand_sharpness": 1e-05,
      "pinn_expand_time_base": "local"
    },
    "effective_training_env": {
      "CASE": "drift",
      "DRIFT_DT": "0.1",
      "DRIFT_RESET_MIXTURE_PROBS": "0.3,0.3,0.4",
      "DRIFT_RESET_MODE": "mixture",
      "DRIFT_RESET_SCALE": "1.0",
      "DRIFT_RESET_T_MAX": "5.0",
      "DRIFT_RESET_T_MIN": "0.0",
      "DRIFT_RESET_T_MODE": "random",
      "HJB_LAPLACIAN_MODE": "loop",
      "INITIAL_EXPLORATION_POLICY": "policy",
      "METHOD": "scheduling",
      "NUM_WORKERS": "2",
      "PINN_EXPAND_CENTER": "500000",
      "PINN_EXPAND_JITTER_FINAL": "0.01",
      "PINN_EXPAND_JITTER_INITIAL": "0.01",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "1.0",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1198,
        "max_abs_mc_v": 0.9978,
        "mean_mc": 0.4089,
        "mean_v": 0.2894
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1188,
        "max_abs_mc_v": 0.7873,
        "mean_mc": 0.6014,
        "mean_v": 0.4864
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.006363322027027607,
      "Loss/HJB": 0.6146804094314575,
      "Loss/HJB_replay": 1.783735990524292,
      "Loss/BDR": 0.28107231855392456,
      "Weights/RL": 1.0,
      "Weights/HJB": 9.999999747378752e-05,
      "Weights/BDR": 9.999999747378752e-05,
      "PINN/Replay Jitter": 0.009999999776482582
    }
  },
  {
    "name": "ramp0001to00025_expand001_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_000/ramp0to0001_expand001/train/0704_1419_ramp0to0001_expand001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_001/ramp0001to00025_expand001_R1/train/0705_0100_ramp0001to00025_expand001_R1_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "ramp0001to00025_expand001_R1",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_5Mto10M/round_000/ramp0to0001_expand001/train/0704_1419_ramp0to0001_expand001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.0001,
        0.0001
      ],
      "schedule_final": [
        1.0,
        0.00025,
        0.00025
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.01,
      "pinn_expand_jitter_final": 0.01,
      "pinn_expand_center": 500000,
      "pinn_expand_sharpness": 1e-05,
      "pinn_expand_time_base": "local"
    },
    "effective_training_env": {
      "CASE": "drift",
      "DRIFT_DT": "0.1",
      "DRIFT_RESET_MIXTURE_PROBS": "0.3,0.3,0.4",
      "DRIFT_RESET_MODE": "mixture",
      "DRIFT_RESET_SCALE": "1.0",
      "DRIFT_RESET_T_MAX": "5.0",
      "DRIFT_RESET_T_MIN": "0.0",
      "DRIFT_RESET_T_MODE": "random",
      "HJB_LAPLACIAN_MODE": "loop",
      "INITIAL_EXPLORATION_POLICY": "policy",
      "METHOD": "scheduling",
      "NUM_WORKERS": "2",
      "PINN_EXPAND_CENTER": "500000",
      "PINN_EXPAND_JITTER_FINAL": "0.01",
      "PINN_EXPAND_JITTER_INITIAL": "0.01",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "1.0",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1282,
        "max_abs_mc_v": 0.9981,
        "mean_mc": 0.4143,
        "mean_v": 0.2863
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1216,
        "max_abs_mc_v": 0.8543,
        "mean_mc": 0.597,
        "mean_v": 0.4791
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.0074008191004395485,
      "Loss/HJB": 0.5209445953369141,
      "Loss/HJB_replay": 1.4289565086364746,
      "Loss/BDR": 0.29581379890441895,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0002500000118743628,
      "Weights/BDR": 0.0002500000118743628,
      "PINN/Replay Jitter": 0.009999999776482582
    }
  }
]
