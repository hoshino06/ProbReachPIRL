You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_004_plan.json
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
      "schedule_sharpness": 1e-5
    }
  ]
}

Selection rules:
- Continue from the best safe checkpoint when reward and MC are stable.
- If reward or meanMC degraded, reduce weights or slow the schedule before trying larger weights.
- Increase HJB/BDR gradually.
- Use at most one TD3-restart control per round, unless all scheduling checkpoints collapsed.
- Do not repeat an existing start_checkpoint + schedule_initial + schedule_final combination unless round_note explains why.

Advisor context from TOML:

Manual notes:
- This is a restart after fixed2randT_replayHJB_5Mto10M collapsed reward in round0/round1.
- The fixedT TD3 5M baseline had RL/Average Reward around 0.44 at the end. Treat reward below 0.20 as unsafe and below 0.10 as collapsed, even if HJB/BDR or MC metrics improve.
- Do not continue, back off from, or branch from collapsed checkpoints. If all candidates in a round are unsafe, restart from the baseline checkpoint or the last checkpoint with reward >= 0.20.
- Do not increase HJB/BDR above 0.005 until reward stays at least 0.30 for a full round. Prefer smaller increases such as 0.001 -> 0.002 -> 0.005.
- Keep one randT reset control with HJB/BDR=0.0 until the reward effect of switching fixedT to randT is understood.
- Prioritize reward preservation over reducing HJB loss. MC reachability improvements are not useful if reward collapses.

Reference paths:
- logs/drift/README.md
- logs/drift/scheduling_randT_replayHJB
- logs/drift/scheduling_fixed2randT_uniformHJB
- logs/drift/scheduling_fixedT_replayHJB
- logs/drift/scheduling_fixedT_uniformHJB
- logs/drift/td3_T01_randT/
- logs/drift/td3_T01/
- scheduling_experiment/fixed2randT_replayHJB_5Mto10M


Completed results JSON:
[
  {
    "name": "randTonly",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/randTonly/train/0626_1710_randTonly_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "randTonly",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1359,
        "max_abs_mc_v": 0.9993,
        "mean_mc": 0.4007,
        "mean_v": 0.2661
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1308,
        "max_abs_mc_v": 0.8659,
        "mean_mc": 0.5922,
        "mean_v": 0.4661
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.005866708233952522,
      "Loss/HJB": 1.0787588357925415,
      "Loss/HJB_replay": 2.318056583404541,
      "Loss/BDR": 0.3721981346607208,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0,
      "Weights/BDR": 0.0
    }
  },
  {
    "name": "ramp0to0001",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "ramp0to0001",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.001,
        0.001
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1174,
        "max_abs_mc_v": 0.9978,
        "mean_mc": 0.4056,
        "mean_v": 0.2888
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.107,
        "max_abs_mc_v": 0.8076,
        "mean_mc": 0.5878,
        "mean_v": 0.4891
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.2199999988079071,
      "Loss/RL": 0.004622244276106358,
      "Loss/HJB": 0.3693614602088928,
      "Loss/HJB_replay": 0.2322288602590561,
      "Loss/BDR": 0.1754508912563324,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0010000000474974513,
      "Weights/BDR": 0.0010000000474974513
    }
  },
  {
    "name": "hold0001",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_001/hold0001/train/0627_0211_hold0001_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "hold0001",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.001,
        0.001
      ],
      "schedule_final": [
        1.0,
        0.001,
        0.001
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1148,
        "max_abs_mc_v": 0.9947,
        "mean_mc": 0.413,
        "mean_v": 0.2986
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.102,
        "max_abs_mc_v": 0.7782,
        "mean_mc": 0.5886,
        "mean_v": 0.4925
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1899999976158142,
      "Loss/RL": 0.002954805037006736,
      "Loss/HJB": 0.7230154275894165,
      "Loss/HJB_replay": 0.375631183385849,
      "Loss/BDR": 0.14958958327770233,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0010000000474974513,
      "Weights/BDR": 0.0010000000474974513
    }
  },
  {
    "name": "cont0001to0002",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_001/cont0001to0002/train/0627_0211_cont0001to0002_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "cont0001to0002",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.001,
        0.001
      ],
      "schedule_final": [
        1.0,
        0.002,
        0.002
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1244,
        "max_abs_mc_v": 0.9956,
        "mean_mc": 0.4142,
        "mean_v": 0.2902
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1008,
        "max_abs_mc_v": 0.8494,
        "mean_mc": 0.5812,
        "mean_v": 0.4851
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1899999976158142,
      "Loss/RL": 0.01339755766093731,
      "Loss/HJB": 0.7431148290634155,
      "Loss/HJB_replay": 0.3475741744041443,
      "Loss/BDR": 0.13531306385993958,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0020000000949949026,
      "Weights/BDR": 0.0020000000949949026
    }
  },
  {
    "name": "backoff0005",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_002/backoff0005/train/0627_1217_backoff0005_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "backoff0005",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.001,
        0.001
      ],
      "schedule_final": [
        1.0,
        0.0005,
        0.0005
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1178,
        "max_abs_mc_v": 0.9965,
        "mean_mc": 0.4125,
        "mean_v": 0.2952
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0975,
        "max_abs_mc_v": 0.8352,
        "mean_mc": 0.5849,
        "mean_v": 0.4942
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1899999976158142,
      "Loss/RL": 0.007639835588634014,
      "Loss/HJB": 0.9151854515075684,
      "Loss/HJB_replay": 0.5165854692459106,
      "Loss/BDR": 0.2724255323410034,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0005000000237487257,
      "Weights/BDR": 0.0005000000237487257
    }
  },
  {
    "name": "recover0000",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_002/recover0000/train/0627_1217_recover0000_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "recover0000",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.001,
        0.001
      ],
      "schedule_final": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1271,
        "max_abs_mc_v": 0.9982,
        "mean_mc": 0.412,
        "mean_v": 0.2851
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.119,
        "max_abs_mc_v": 0.8754,
        "mean_mc": 0.5982,
        "mean_v": 0.4826
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.003835519077256322,
      "Loss/HJB": 0.6718581914901733,
      "Loss/HJB_replay": 1.7724525928497314,
      "Loss/BDR": 0.31113511323928833,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0,
      "Weights/BDR": 0.0
    }
  },
  {
    "name": "backoff00025",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_003/backoff00025/train/0627_2221_backoff00025_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "backoff00025",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_000/ramp0to0001/train/0626_1710_ramp0to0001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.001,
        0.001
      ],
      "schedule_final": [
        1.0,
        0.00025,
        0.00025
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1239,
        "max_abs_mc_v": 0.9988,
        "mean_mc": 0.4173,
        "mean_v": 0.2936
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1095,
        "max_abs_mc_v": 0.9203,
        "mean_mc": 0.5943,
        "mean_v": 0.4889
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.003060895251110196,
      "Loss/HJB": 1.8402177095413208,
      "Loss/HJB_replay": 0.6877301931381226,
      "Loss/BDR": 0.18272070586681366,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0002500000118743628,
      "Weights/BDR": 0.0002500000118743628
    }
  },
  {
    "name": "restart0005",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_003/restart0005/train/0627_2221_restart0005_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "restart0005",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.0005,
        0.0005
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1119,
        "max_abs_mc_v": 0.998,
        "mean_mc": 0.3954,
        "mean_v": 0.2876
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1044,
        "max_abs_mc_v": 0.8605,
        "mean_mc": 0.5762,
        "mean_v": 0.4862
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.006910892203450203,
      "Loss/HJB": 0.6671096086502075,
      "Loss/HJB_replay": 0.3602061867713928,
      "Loss/BDR": 0.19598865509033203,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0005000000237487257,
      "Weights/BDR": 0.0005000000237487257
    }
  }
]
