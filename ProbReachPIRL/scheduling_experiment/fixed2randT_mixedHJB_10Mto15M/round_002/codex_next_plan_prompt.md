You are controlling the next round of PIRL weight and collocation-distribution scheduling.

Objective:
- Treat 15000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.
- For replay_expand experiments, start close to replay-HJB and expand the HJB collocation neighborhood cautiously.

Output:
- Write ONLY valid JSON to: /home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_003_plan.json
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
- This experiment starts from the same 10M checkpoint as fixed2randT_replayExpandHJB_10Mto15M and again targets 15M.
- The prior 10M-to-15M replay_expand run preserved MC reachability, but Loss/HJB on uniform collocation stayed high relative to Loss/HJB_replay.
- Primary objective: reduce uniform-sampled Loss/HJB while preserving MC reachability. Treat Loss/HJB_replay as secondary; a lower replay loss alone is not success.
- Guardrails: keep beta-r meanMC near or above 0.41 and ey-epsi meanMC near or above 0.58 when possible. If both meanMC values drop materially, back off uniform fraction, jitter, or HJB/BDR weights.
- Prefer changing the collocation distribution before increasing HJB/BDR weights. The first lever is PINN_REPLAY_FRACTION below 1.0, which mixes uniform PDE points into replay_expand training.
- Interpret PINN_REPLAY_FRACTION as replay PDE fraction: 0.90 means 10% uniform, 0.75 means 25% uniform.
- Keep HJB/BDR around 0.0015 for the first round so the effect of adding uniform collocation is identifiable.
- Increase uniform fraction gradually: try 10-25% uniform first; do not jump below 0.5 replay fraction unless MC remains stable.
- Use replay_expand jitter only as local thickening. Jitter 0.0125-0.02 was too local to lower global HJB reliably; try 0.02-0.05 cautiously, but avoid sacrificing MC for lower uniform Loss/HJB.
- Candidate selection should rank runs by: (1) no MC collapse, (2) lower Loss/HJB, (3) acceptable mean|MC-V|, (4) stable reward.
- Compare against fixed2randT_replayExpandHJB_10Mto15M/all_results.json, especially the 15M hold0015_expand00125_R5 result: beta-r meanMC 0.4194, ey-epsi meanMC 0.5858, uniform Loss/HJB 0.5676.

Reference paths:
- logs/drift/README.md
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2.toml
- scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/all_results.json
- scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M.toml
- scheduling_experiment/fixed2randT_replayExpandHJB_10Mto15M/all_results.json
- scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/all_results.json


Completed results JSON:
[
  {
    "name": "mix90_expand003_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000",
    "start_itr": 10000000,
    "checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_000/mix90_expand003_R1/train/0706_2022_mix90_expand003_R1_seed_1/ckpt-11000000",
    "target_updates": 11000000,
    "candidate": {
      "name": "mix90_expand003_R1",
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
      "pinn_replay_fraction": 0.9,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.02,
      "pinn_expand_jitter_final": 0.03,
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
      "PINN_EXPAND_JITTER_FINAL": "0.03",
      "PINN_EXPAND_JITTER_INITIAL": "0.02",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "0.9",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta-r": {
        "mean_abs_mc_v": 0.11741010610195142,
        "max_abs_mc_v": 0.9937953711487353,
        "mean_mc": 0.4159079084287201,
        "mean_v": 0.3010053667527766
      },
      "ey-epsi": {
        "mean_abs_mc_v": 0.09538615739535813,
        "max_abs_mc_v": 0.8145179562270641,
        "mean_mc": 0.5722554630593132,
        "mean_v": 0.48958381589039085
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.17000000178813934,
      "Loss/RL": 0.006836882792413235,
      "Loss/HJB": 0.1984332799911499,
      "Loss/HJB_replay": 0.37216857075691223,
      "Loss/BDR": 0.05943482369184494,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.029999999329447746
    },
    "recovered": true,
    "recovery_note": "Recovered from completed MC npz files after plot_drift_mc_reachability processes hung before subprocess return."
  },
  {
    "name": "mix75_expand002_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000",
    "start_itr": 10000000,
    "checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_000/mix75_expand002_R1/train/0706_2022_mix75_expand002_R1_seed_1/ckpt-11000000",
    "target_updates": 11000000,
    "candidate": {
      "name": "mix75_expand002_R1",
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
      "pinn_replay_fraction": 0.75,
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
      "PINN_REPLAY_FRACTION": "0.75",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta-r": {
        "mean_abs_mc_v": 0.10947346967729091,
        "max_abs_mc_v": 0.9941926929168403,
        "mean_mc": 0.40244536940686787,
        "mean_v": 0.29989399338442024
      },
      "ey-epsi": {
        "mean_abs_mc_v": 0.09531718482424473,
        "max_abs_mc_v": 0.819036528468132,
        "mean_mc": 0.5485171696149844,
        "mean_v": 0.48673099482541365
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.002989153377711773,
      "Loss/HJB": 0.18761442601680756,
      "Loss/HJB_replay": 0.306663453578949,
      "Loss/BDR": 0.10632450878620148,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.019999999552965164
    },
    "recovered": true,
    "recovery_note": "Recovered from completed MC npz files after plot_drift_mc_reachability processes hung before subprocess return."
  },
  {
    "name": "mix90_expand0035_R2",
    "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_000/mix90_expand003_R1/train/0706_2022_mix90_expand003_R1_seed_1/ckpt-11000000",
    "start_itr": 11000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_001/mix90_expand0035_R2/train/0707_0901_mix90_expand0035_R2_seed_1/ckpt-12000000",
    "target_updates": 12000000,
    "candidate": {
      "name": "mix90_expand0035_R2",
      "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_000/mix90_expand003_R1/train/0706_2022_mix90_expand003_R1_seed_1/ckpt-11000000",
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
      "pinn_replay_fraction": 0.9,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.03,
      "pinn_expand_jitter_final": 0.035,
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
      "PINN_EXPAND_JITTER_FINAL": "0.035",
      "PINN_EXPAND_JITTER_INITIAL": "0.03",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "0.9",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1067,
        "max_abs_mc_v": 0.997,
        "mean_mc": 0.4065,
        "mean_v": 0.3048
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0945,
        "max_abs_mc_v": 0.8421,
        "mean_mc": 0.5711,
        "mean_v": 0.4947
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.00650983490049839,
      "Loss/HJB": 0.19288897514343262,
      "Loss/HJB_replay": 0.2838706970214844,
      "Loss/BDR": 0.08086375147104263,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.03500000014901161
    }
  },
  {
    "name": "mix85_expand003_R2",
    "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_000/mix90_expand003_R1/train/0706_2022_mix90_expand003_R1_seed_1/ckpt-11000000",
    "start_itr": 11000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_001/mix85_expand003_R2/train/0707_0901_mix85_expand003_R2_seed_1/ckpt-12000000",
    "target_updates": 12000000,
    "candidate": {
      "name": "mix85_expand003_R2",
      "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_000/mix90_expand003_R1/train/0706_2022_mix90_expand003_R1_seed_1/ckpt-11000000",
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
      "pinn_replay_fraction": 0.85,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.025,
      "pinn_expand_jitter_final": 0.03,
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
      "PINN_EXPAND_JITTER_FINAL": "0.03",
      "PINN_EXPAND_JITTER_INITIAL": "0.025",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "0.85",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1168,
        "max_abs_mc_v": 0.9968,
        "mean_mc": 0.4207,
        "mean_v": 0.3046
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0994,
        "max_abs_mc_v": 0.7884,
        "mean_mc": 0.5901,
        "mean_v": 0.4982
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.005607063416391611,
      "Loss/HJB": 0.2479092925786972,
      "Loss/HJB_replay": 0.2894335091114044,
      "Loss/BDR": 0.09039280563592911,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.029999999329447746
    }
  },
  {
    "name": "mix80_expand003_R3",
    "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_001/mix85_expand003_R2/train/0707_0901_mix85_expand003_R2_seed_1/ckpt-12000000",
    "start_itr": 12000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_002/mix80_expand003_R3/train/0707_2126_mix80_expand003_R3_seed_1/ckpt-13000000",
    "target_updates": 13000000,
    "candidate": {
      "name": "mix80_expand003_R3",
      "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_001/mix85_expand003_R2/train/0707_0901_mix85_expand003_R2_seed_1/ckpt-12000000",
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
      "pinn_replay_fraction": 0.8,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.03,
      "pinn_expand_jitter_final": 0.03,
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
      "PINN_EXPAND_JITTER_FINAL": "0.03",
      "PINN_EXPAND_JITTER_INITIAL": "0.03",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "0.8",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1066,
        "max_abs_mc_v": 0.994,
        "mean_mc": 0.4138,
        "mean_v": 0.3106
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0866,
        "max_abs_mc_v": 0.7468,
        "mean_mc": 0.5795,
        "mean_v": 0.5084
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.17000000178813934,
      "Loss/RL": 0.008122154511511326,
      "Loss/HJB": 0.19658461213111877,
      "Loss/HJB_replay": 0.3141019940376282,
      "Loss/BDR": 0.06503298878669739,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.029999999329447746
    }
  },
  {
    "name": "mix90_expand003_back_R3",
    "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_001/mix90_expand0035_R2/train/0707_0901_mix90_expand0035_R2_seed_1/ckpt-12000000",
    "start_itr": 12000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_002/mix90_expand003_back_R3/train/0707_2126_mix90_expand003_back_R3_seed_1/ckpt-13000000",
    "target_updates": 13000000,
    "candidate": {
      "name": "mix90_expand003_back_R3",
      "start_checkpoint": "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_001/mix90_expand0035_R2/train/0707_0901_mix90_expand0035_R2_seed_1/ckpt-12000000",
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
      "pinn_replay_fraction": 0.9,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.035,
      "pinn_expand_jitter_final": 0.03,
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
      "PINN_EXPAND_JITTER_FINAL": "0.03",
      "PINN_EXPAND_JITTER_INITIAL": "0.035",
      "PINN_EXPAND_SHARPNESS": "1e-05",
      "PINN_EXPAND_TIME_BASE": "local",
      "PINN_REPLAY_FRACTION": "0.9",
      "PINN_REPLAY_JITTER": "0.0",
      "PINN_SAMPLE_MODE": "replay_expand",
      "SCHEDULE_TIME_BASE": "local",
      "SEEDS": "1"
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1165,
        "max_abs_mc_v": 0.994,
        "mean_mc": 0.4193,
        "mean_v": 0.3033
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1004,
        "max_abs_mc_v": 0.9141,
        "mean_mc": 0.591,
        "mean_v": 0.4988
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.20000000298023224,
      "Loss/RL": 0.0049653202295303345,
      "Loss/HJB": 0.11546194553375244,
      "Loss/HJB_replay": 0.3950135111808777,
      "Loss/BDR": 0.13489416241645813,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.001500000013038516,
      "Weights/BDR": 0.001500000013038516,
      "PINN/Replay Jitter": 0.029999999329447746
    }
  }
]
