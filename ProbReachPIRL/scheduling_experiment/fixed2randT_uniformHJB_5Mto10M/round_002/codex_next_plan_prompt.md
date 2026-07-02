You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003_plan.json
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
- This matches manual fixed2randT_uniformHJB: fixedT TD3 baseline, randT scheduling reset, uniform HJB samples.
- Compare against fixed2randT_replayHJB to isolate the HJB sampling distribution.
- HJB/BDR=0.05 is a useful manual-style comparison point, but back off if reward or MC reachability degrades.

Reference paths:
- logs/drift/README.md
- logs/drift/scheduling_randT_replayHJB
- logs/drift/scheduling_fixed2randT_uniformHJB
- logs/drift/scheduling_fixedT_replayHJB
- logs/drift/scheduling_fixedT_uniformHJB
- logs/drift/td3_T01_randT/
- logs/drift/td3_T01/


Completed results JSON:
[
  {
    "name": "ramp0to001",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_000/ramp0to001/train/0625_1015_ramp0to001_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "ramp0to001",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.01,
        0.01
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1695,
        "max_abs_mc_v": 0.9974,
        "mean_mc": 0.4047,
        "mean_v": 0.2359
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1675,
        "max_abs_mc_v": 0.8948,
        "mean_mc": 0.5756,
        "mean_v": 0.4117
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.004678318277001381,
      "Loss/HJB": 0.05425982177257538,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.011371891014277935,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp0to005",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_000/ramp0to005/train/0625_1015_ramp0to005_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "ramp0to005",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.05,
        0.05
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1608,
        "max_abs_mc_v": 0.9852,
        "mean_mc": 0.3743,
        "mean_v": 0.2151
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1633,
        "max_abs_mc_v": 0.8537,
        "mean_mc": 0.5672,
        "mean_v": 0.4078
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.005194326862692833,
      "Loss/HJB": 0.007039847318083048,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.011740513145923615,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.05000000074505806,
      "Weights/BDR": 0.05000000074505806
    }
  },
  {
    "name": "hold005_from6M",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_000/ramp0to005/train/0625_1015_ramp0to005_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_001/hold005_from6M/train/0626_0905_hold005_from6M_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "hold005_from6M",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_000/ramp0to005/train/0625_1015_ramp0to005_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.05,
        0.05
      ],
      "schedule_final": [
        1.0,
        0.05,
        0.05
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.2069,
        "max_abs_mc_v": 0.9986,
        "mean_mc": 0.3954,
        "mean_v": 0.1899
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1694,
        "max_abs_mc_v": 0.9691,
        "mean_mc": 0.5676,
        "mean_v": 0.4011
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.004775560926645994,
      "Loss/HJB": 0.010396385565400124,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.009838419035077095,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.05000000074505806,
      "Weights/BDR": 0.05000000074505806
    }
  },
  {
    "name": "ramp005to0075",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_000/ramp0to005/train/0625_1015_ramp0to005_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_001/ramp005to0075/train/0626_0905_ramp005to0075_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "ramp005to0075",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_000/ramp0to005/train/0625_1015_ramp0to005_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.05,
        0.05
      ],
      "schedule_final": [
        1.0,
        0.075,
        0.075
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.2068,
        "max_abs_mc_v": 0.9991,
        "mean_mc": 0.3814,
        "mean_v": 0.176
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.18,
        "max_abs_mc_v": 0.9674,
        "mean_mc": 0.5708,
        "mean_v": 0.3945
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10999999940395355,
      "Loss/RL": 0.004913056269288063,
      "Loss/HJB": 0.0026158469263464212,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.008630027063190937,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.07500000298023224,
      "Weights/BDR": 0.07500000298023224
    }
  },
  {
    "name": "backoff005to0025_from7M",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_001/hold005_from6M/train/0626_0905_hold005_from6M_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_002/backoff005to0025_from7M/train/0627_0137_backoff005to0025_from7M_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "backoff005to0025_from7M",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_001/hold005_from6M/train/0626_0905_hold005_from6M_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.05,
        0.05
      ],
      "schedule_final": [
        1.0,
        0.025,
        0.025
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.2233,
        "max_abs_mc_v": 0.9992,
        "mean_mc": 0.3967,
        "mean_v": 0.1739
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1964,
        "max_abs_mc_v": 0.9751,
        "mean_mc": 0.5589,
        "mean_v": 0.3656
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.009233697317540646,
      "Loss/HJB": 0.00929327867925167,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.023045208305120468,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.02500000037252903,
      "Weights/BDR": 0.02500000037252903
    }
  },
  {
    "name": "restart_ramp0to0025",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_002/restart_ramp0to0025/train/0627_0137_restart_ramp0to0025_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "restart_ramp0to0025",
      "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.025,
        0.025
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1242,
        "max_abs_mc_v": 0.9967,
        "mean_mc": 0.3556,
        "mean_v": 0.2408
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1311,
        "max_abs_mc_v": 0.9197,
        "mean_mc": 0.5051,
        "mean_v": 0.4246
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.004878987092524767,
      "Loss/HJB": 0.013665672391653061,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.018612805753946304,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.02500000037252903,
      "Weights/BDR": 0.02500000037252903
    }
  }
]
