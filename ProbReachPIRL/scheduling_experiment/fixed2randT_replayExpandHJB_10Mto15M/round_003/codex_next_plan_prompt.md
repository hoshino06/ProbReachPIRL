You are controlling the next round of PIRL weight and collocation-distribution scheduling.

Objective:
- Treat 15000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.
- For replay_expand experiments, start close to replay-HJB and expand the HJB collocation neighborhood cautiously.

Output:
- Write ONLY valid JSON to: /home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_004_plan.json
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
- This is a local continuation experiment from the strongest 10M fixed2randT_replayHJB_restart2 checkpoint.
- The 10M hold0015_R3 checkpoint had good reward and the best beta-r meanMC among the restart2 10M checkpoints; ramp00175_R3 had lower HJB/BDR losses but slightly worse beta-r calibration.
- The goal is not to chase lower uniform HJB loss aggressively. First preserve MC reachability and reward, then widen the HJB collocation distribution.
- Treat PINN/Replay Jitter like a loss weight: hold or back off jitter when reward, beta-r meanMC, or ey-epsi meanMC weakens.
- Prefer small jitter finals such as 0.005, 0.01, 0.02, or 0.03 before trying wider neighborhoods.
- Keep HJB/BDR near the successful replay-HJB range around 0.0015 unless the run remains stable for a full round.
- Compare every branch against pure replay-HJB restart2 continuation around 10M to 12M before interpreting lower Loss/HJB as an improvement.

Reference paths:
- logs/drift/README.md
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2.toml
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/all_results.json
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/results.json
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_003/results.json
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_004/results.json
- scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/all_results.json


Completed results JSON:
[
  {
    "name": "hold0015_expand001_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000",
    "start_itr": 10000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_000/hold0015_expand001_R1/train/0703_2134_hold0015_expand001_R1_seed_1/ckpt-11000000",
    "target_updates": 11000000,
    "candidate": {
      "name": "hold0015_expand001_R1",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
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
        "mean_abs_mc_v": 0.1248,
        "max_abs_mc_v": 0.9968,
        "mean_mc": 0.4261,
        "mean_v": 0.3016
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0974,
        "max_abs_mc_v": 0.7788,
        "mean_mc": 0.5917,
        "mean_v": 0.5013
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.11999999731779099,
      "Loss/RL": 0.00399504229426384,
      "Loss/HJB": 0.6230176091194153,
      "Loss/HJB_replay": 0.2607784867286682,
      "Loss/BDR": 0.0484834648668766,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.009999999776482582
    }
  },
  {
    "name": "hold0015_expand002_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000",
    "start_itr": 10000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_000/hold0015_expand002_R1/train/0703_2134_hold0015_expand002_R1_seed_1/ckpt-11000000",
    "target_updates": 11000000,
    "candidate": {
      "name": "hold0015_expand002_R1",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.0,
      "pinn_expand_jitter_final": 0.02,
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
      "PINN_EXPAND_JITTER_FINAL": "0.02",
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
        "mean_abs_mc_v": 0.1122,
        "max_abs_mc_v": 0.9964,
        "mean_mc": 0.4185,
        "mean_v": 0.3079
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0976,
        "max_abs_mc_v": 0.8086,
        "mean_mc": 0.5762,
        "mean_v": 0.4948
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.23000000417232513,
      "Loss/RL": 0.005927258636802435,
      "Loss/HJB": 0.34248965978622437,
      "Loss/HJB_replay": 0.16739392280578613,
      "Loss/BDR": 0.07731422036886215,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.019999999552965164
    }
  },
  {
    "name": "hold0015_expand002_hold_R2",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_000/hold0015_expand002_R1/train/0703_2134_hold0015_expand002_R1_seed_1/ckpt-11000000",
    "start_itr": 11000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_001/hold0015_expand002_hold_R2/train/0704_1018_hold0015_expand002_hold_R2_seed_1/ckpt-12000000",
    "target_updates": 12000000,
    "candidate": {
      "name": "hold0015_expand002_hold_R2",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_000/hold0015_expand002_R1/train/0703_2134_hold0015_expand002_R1_seed_1/ckpt-11000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.02,
      "pinn_expand_jitter_final": 0.02,
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
      "PINN_EXPAND_JITTER_FINAL": "0.02",
      "PINN_EXPAND_JITTER_INITIAL": "0.02",
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
        "mean_abs_mc_v": 0.1112,
        "max_abs_mc_v": 0.9923,
        "mean_mc": 0.4166,
        "mean_v": 0.3074
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0942,
        "max_abs_mc_v": 0.8981,
        "mean_mc": 0.5829,
        "mean_v": 0.5026
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.004685189109295607,
      "Loss/HJB": 0.9557195901870728,
      "Loss/HJB_replay": 0.3971715271472931,
      "Loss/BDR": 0.08018558472394943,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.019999999552965164
    }
  },
  {
    "name": "hold0015_expand0015_R2",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_000/hold0015_expand002_R1/train/0703_2134_hold0015_expand002_R1_seed_1/ckpt-11000000",
    "start_itr": 11000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_001/hold0015_expand0015_R2/train/0704_1018_hold0015_expand0015_R2_seed_1/ckpt-12000000",
    "target_updates": 12000000,
    "candidate": {
      "name": "hold0015_expand0015_R2",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_000/hold0015_expand002_R1/train/0703_2134_hold0015_expand002_R1_seed_1/ckpt-11000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.02,
      "pinn_expand_jitter_final": 0.015,
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
      "PINN_EXPAND_JITTER_FINAL": "0.015",
      "PINN_EXPAND_JITTER_INITIAL": "0.02",
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
        "mean_abs_mc_v": 0.1154,
        "max_abs_mc_v": 0.9978,
        "mean_mc": 0.4205,
        "mean_v": 0.3059
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0974,
        "max_abs_mc_v": 0.7982,
        "mean_mc": 0.5918,
        "mean_v": 0.5033
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.23999999463558197,
      "Loss/RL": 0.0053792595863342285,
      "Loss/HJB": 0.3918224573135376,
      "Loss/HJB_replay": 0.2967182397842407,
      "Loss/BDR": 0.06918779760599136,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.014999999664723873
    }
  },
  {
    "name": "hold0015_expand0015_hold_R3",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_001/hold0015_expand0015_R2/train/0704_1018_hold0015_expand0015_R2_seed_1/ckpt-12000000",
    "start_itr": 12000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_002/hold0015_expand0015_hold_R3/train/0704_2240_hold0015_expand0015_hold_R3_seed_1/ckpt-13000000",
    "target_updates": 13000000,
    "candidate": {
      "name": "hold0015_expand0015_hold_R3",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_001/hold0015_expand0015_R2/train/0704_1018_hold0015_expand0015_R2_seed_1/ckpt-12000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.015,
      "pinn_expand_jitter_final": 0.015,
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
      "PINN_EXPAND_JITTER_FINAL": "0.015",
      "PINN_EXPAND_JITTER_INITIAL": "0.015",
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
        "mean_abs_mc_v": 0.1159,
        "max_abs_mc_v": 0.9963,
        "mean_mc": 0.4202,
        "mean_v": 0.3057
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0973,
        "max_abs_mc_v": 0.8155,
        "mean_mc": 0.5897,
        "mean_v": 0.5003
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1899999976158142,
      "Loss/RL": 0.004288747441023588,
      "Loss/HJB": 0.41682982444763184,
      "Loss/HJB_replay": 0.2126314640045166,
      "Loss/BDR": 0.11781341582536697,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.014999999664723873
    }
  },
  {
    "name": "hold0015_expand002_R3",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_001/hold0015_expand0015_R2/train/0704_1018_hold0015_expand0015_R2_seed_1/ckpt-12000000",
    "start_itr": 12000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_002/hold0015_expand002_R3/train/0704_2240_hold0015_expand002_R3_seed_1/ckpt-13000000",
    "target_updates": 13000000,
    "candidate": {
      "name": "hold0015_expand002_R3",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_001/hold0015_expand0015_R2/train/0704_1018_hold0015_expand0015_R2_seed_1/ckpt-12000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.015,
      "pinn_expand_jitter_final": 0.02,
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
      "PINN_EXPAND_JITTER_FINAL": "0.02",
      "PINN_EXPAND_JITTER_INITIAL": "0.015",
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
        "mean_abs_mc_v": 0.1073,
        "max_abs_mc_v": 0.9959,
        "mean_mc": 0.4075,
        "mean_v": 0.3043
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0958,
        "max_abs_mc_v": 0.8817,
        "mean_mc": 0.5778,
        "mean_v": 0.4968
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.17000000178813934,
      "Loss/RL": 0.005790249444544315,
      "Loss/HJB": 0.5555553436279297,
      "Loss/HJB_replay": 0.2408113181591034,
      "Loss/BDR": 0.09967690706253052,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.019999999552965164
    }
  },
  {
    "name": "hold0015_expand0015_hold_R4",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_002/hold0015_expand0015_hold_R3/train/0704_2240_hold0015_expand0015_hold_R3_seed_1/ckpt-13000000",
    "start_itr": 13000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_003/hold0015_expand0015_hold_R4/train/0705_1103_hold0015_expand0015_hold_R4_seed_1/ckpt-14000000",
    "target_updates": 14000000,
    "candidate": {
      "name": "hold0015_expand0015_hold_R4",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_002/hold0015_expand0015_hold_R3/train/0704_2240_hold0015_expand0015_hold_R3_seed_1/ckpt-13000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.015,
      "pinn_expand_jitter_final": 0.015,
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
      "PINN_EXPAND_JITTER_FINAL": "0.015",
      "PINN_EXPAND_JITTER_INITIAL": "0.015",
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
        "mean_abs_mc_v": 0.1129,
        "max_abs_mc_v": 0.996,
        "mean_mc": 0.419,
        "mean_v": 0.3071
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0949,
        "max_abs_mc_v": 0.8105,
        "mean_mc": 0.5928,
        "mean_v": 0.5074
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.17000000178813934,
      "Loss/RL": 0.008607110008597374,
      "Loss/HJB": 0.490619421005249,
      "Loss/HJB_replay": 0.22703680396080017,
      "Loss/BDR": 0.0849749743938446,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.014999999664723873
    }
  },
  {
    "name": "hold0015_expand00175_R4",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_002/hold0015_expand0015_hold_R3/train/0704_2240_hold0015_expand0015_hold_R3_seed_1/ckpt-13000000",
    "start_itr": 13000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_003/hold0015_expand00175_R4/train/0705_1103_hold0015_expand00175_R4_seed_1/ckpt-14000000",
    "target_updates": 14000000,
    "candidate": {
      "name": "hold0015_expand00175_R4",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/round_002/hold0015_expand0015_hold_R3/train/0704_2240_hold0015_expand0015_hold_R3_seed_1/ckpt-13000000",
      "schedule_initial": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_final": [
        1.0,
        0.0015,
        0.0015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.015,
      "pinn_expand_jitter_final": 0.0175,
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
      "PINN_EXPAND_JITTER_FINAL": "0.0175",
      "PINN_EXPAND_JITTER_INITIAL": "0.015",
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
        "mean_abs_mc_v": 0.104,
        "max_abs_mc_v": 0.9962,
        "mean_mc": 0.4161,
        "mean_v": 0.3139
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0882,
        "max_abs_mc_v": 0.8916,
        "mean_mc": 0.584,
        "mean_v": 0.5074
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.002462746575474739,
      "Loss/HJB": 0.6841161251068115,
      "Loss/HJB_replay": 0.24855133891105652,
      "Loss/BDR": 0.12907779216766357,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.017500000074505806
    }
  }
]
