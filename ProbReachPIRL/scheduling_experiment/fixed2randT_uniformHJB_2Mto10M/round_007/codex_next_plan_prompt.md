You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01/up03M_scale10_mix334/ckpt-2000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_008_plan.json
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
- This is a fixed2randT_uniformHJB scheduling study from the early 2M fixedT TD3 checkpoint.
- Start from fixedT TD3, switch reset T to random during scheduling, and use uniform HJB samples.
- Treat 10M total updates as the first milestone, not a hard stop; if training remains healthy, continuing toward 12M is acceptable.
- Compare against randT_replayHJB_2Mto10M to separate early fixedT-to-randT switching from the randT replay-HJB baseline.
- Compare against fixed2randT_replayHJB and fixed2randT_uniformHJB_5Mto10M to isolate the HJB sampling distribution and the earlier 2M start.
- Keep RL weight at 1.0 and back off if reward or meanMC reachability degrades.
- Because the 2M checkpoint is earlier and less stable than 5M, prefer conservative initial HJB/BDR weights before trying 0.05.

Reference paths:
- logs/drift/README.md
- logs/drift/scheduling_randT_replayHJB
- logs/drift/scheduling_fixed2randT_uniformHJB
- logs/drift/scheduling_fixed2randT_replayHJB
- logs/drift/scheduling_fixedT_uniformHJB
- logs/drift/td3_T01_randT/
- logs/drift/td3_T01/
- scheduling_experiment/randT_replayHJB_2Mto10M.toml
- scheduling_experiment/fixed2randT_uniformHJB_5Mto10M.toml


Completed results JSON:
[
  {
    "name": "ramp0to001",
    "start_checkpoint": "logs/drift/td3_T01/up03M_scale10_mix334/ckpt-2000000",
    "start_itr": 2000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_000/ramp0to001/train/0626_0931_ramp0to001_seed_1/ckpt-3000000",
    "target_updates": 3000000,
    "candidate": {
      "name": "ramp0to001",
      "start_checkpoint": "logs/drift/td3_T01/up03M_scale10_mix334/ckpt-2000000",
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
        "mean_abs_mc_v": 0.0988,
        "max_abs_mc_v": 0.9858,
        "mean_mc": 0.2482,
        "mean_v": 0.1504
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.143,
        "max_abs_mc_v": 0.8513,
        "mean_mc": 0.5736,
        "mean_v": 0.4366
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10999999940395355,
      "Loss/RL": 0.006619823165237904,
      "Loss/HJB": 0.009940875694155693,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.07749871909618378,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp0to0005",
    "start_checkpoint": "logs/drift/td3_T01/up03M_scale10_mix334/ckpt-2000000",
    "start_itr": 2000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_000/ramp0to0005/train/0626_0931_ramp0to0005_seed_1/ckpt-3000000",
    "target_updates": 3000000,
    "candidate": {
      "name": "ramp0to0005",
      "start_checkpoint": "logs/drift/td3_T01/up03M_scale10_mix334/ckpt-2000000",
      "schedule_initial": [
        1.0,
        0.0,
        0.0
      ],
      "schedule_final": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0922,
        "max_abs_mc_v": 0.9753,
        "mean_mc": 0.2567,
        "mean_v": 0.1655
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1473,
        "max_abs_mc_v": 0.8723,
        "mean_mc": 0.5878,
        "mean_v": 0.4459
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.009373575448989868,
      "Loss/HJB": 0.030311040580272675,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.0880696251988411,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "hold0005",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_000/ramp0to0005/train/0626_0931_ramp0to0005_seed_1/ckpt-3000000",
    "start_itr": 3000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_001/hold0005/train/0627_0151_hold0005_seed_1/ckpt-4000000",
    "target_updates": 4000000,
    "candidate": {
      "name": "hold0005",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_000/ramp0to0005/train/0626_0931_ramp0to0005_seed_1/ckpt-3000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0909,
        "max_abs_mc_v": 0.9367,
        "mean_mc": 0.2553,
        "mean_v": 0.1654
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1374,
        "max_abs_mc_v": 0.8863,
        "mean_mc": 0.5715,
        "mean_v": 0.4406
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.17000000178813934,
      "Loss/RL": 0.010384229943156242,
      "Loss/HJB": 0.02125496044754982,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.11028614640235901,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "step00075",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_000/ramp0to0005/train/0626_0931_ramp0to0005_seed_1/ckpt-3000000",
    "start_itr": 3000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_001/step00075/train/0627_0151_step00075_seed_1/ckpt-4000000",
    "target_updates": 4000000,
    "candidate": {
      "name": "step00075",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_000/ramp0to0005/train/0626_0931_ramp0to0005_seed_1/ckpt-3000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.0075,
        0.0075
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0953,
        "max_abs_mc_v": 0.9753,
        "mean_mc": 0.2581,
        "mean_v": 0.1638
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1351,
        "max_abs_mc_v": 0.8549,
        "mean_mc": 0.5715,
        "mean_v": 0.4426
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.07000000029802322,
      "Loss/RL": 0.007426549214869738,
      "Loss/HJB": 0.019408533349633217,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.05996306613087654,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.007499999832361937,
      "Weights/BDR": 0.007499999832361937
    }
  },
  {
    "name": "hold0005_5m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_001/hold0005/train/0627_0151_hold0005_seed_1/ckpt-4000000",
    "start_itr": 4000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
    "target_updates": 5000000,
    "candidate": {
      "name": "hold0005_5m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_001/hold0005/train/0627_0151_hold0005_seed_1/ckpt-4000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0965,
        "max_abs_mc_v": 0.9843,
        "mean_mc": 0.2547,
        "mean_v": 0.1589
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1456,
        "max_abs_mc_v": 0.8555,
        "mean_mc": 0.5732,
        "mean_v": 0.4331
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.0076539781875908375,
      "Loss/HJB": 0.034124039113521576,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.052352480590343475,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "step0006",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_001/hold0005/train/0627_0151_hold0005_seed_1/ckpt-4000000",
    "start_itr": 4000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/step0006/train/0627_1840_step0006_seed_1/ckpt-5000000",
    "target_updates": 5000000,
    "candidate": {
      "name": "step0006",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_001/hold0005/train/0627_0151_hold0005_seed_1/ckpt-4000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.006,
        0.006
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0966,
        "max_abs_mc_v": 0.9656,
        "mean_mc": 0.2516,
        "mean_v": 0.1557
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1458,
        "max_abs_mc_v": 0.8314,
        "mean_mc": 0.5695,
        "mean_v": 0.4298
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.003952622413635254,
      "Loss/HJB": 0.036167170852422714,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.0860992893576622,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.006000000052154064,
      "Weights/BDR": 0.006000000052154064
    }
  },
  {
    "name": "hold0005_6m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_003/hold0005_6m/train/0628_1118_hold0005_6m_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "hold0005_6m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1033,
        "max_abs_mc_v": 0.9631,
        "mean_mc": 0.2538,
        "mean_v": 0.1511
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1691,
        "max_abs_mc_v": 0.9453,
        "mean_mc": 0.5691,
        "mean_v": 0.4035
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.0033827186562120914,
      "Loss/HJB": 0.02410038746893406,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.07015480101108551,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "step00055",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_003/step00055/train/0628_1118_step00055_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "step00055",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.0055,
        0.0055
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.098,
        "max_abs_mc_v": 0.9709,
        "mean_mc": 0.2519,
        "mean_v": 0.1544
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1534,
        "max_abs_mc_v": 0.8697,
        "mean_mc": 0.5655,
        "mean_v": 0.4164
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.004769070073962212,
      "Loss/HJB": 0.03256778419017792,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.08852443099021912,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.005499999970197678,
      "Weights/BDR": 0.005499999970197678
    }
  },
  {
    "name": "backoff0005_7m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_003/step00055/train/0628_1118_step00055_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_004/backoff0005_7m/train/0629_0350_backoff0005_7m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "backoff0005_7m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_003/step00055/train/0628_1118_step00055_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.0055,
        0.0055
      ],
      "schedule_final": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1102,
        "max_abs_mc_v": 0.9908,
        "mean_mc": 0.2557,
        "mean_v": 0.1458
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1849,
        "max_abs_mc_v": 0.8376,
        "mean_mc": 0.5646,
        "mean_v": 0.3839
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10000000149011612,
      "Loss/RL": 0.006471508648246527,
      "Loss/HJB": 0.08610915392637253,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.043285805732011795,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "backoff00045_6m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_004/backoff00045_6m/train/0629_0350_backoff00045_6m_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "backoff00045_6m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.0045,
        0.0045
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1058,
        "max_abs_mc_v": 0.9757,
        "mean_mc": 0.2535,
        "mean_v": 0.1484
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1708,
        "max_abs_mc_v": 0.8835,
        "mean_mc": 0.5726,
        "mean_v": 0.4063
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10000000149011612,
      "Loss/RL": 0.003410683013498783,
      "Loss/HJB": 0.01885632798075676,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.0858379453420639,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0044999998062849045,
      "Weights/BDR": 0.0044999998062849045
    }
  },
  {
    "name": "hold00045_6m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/hold00045_6m/train/0629_2024_hold00045_6m_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "hold00045_6m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0045,
        0.0045
      ],
      "schedule_final": [
        1.0,
        0.0045,
        0.0045
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1106,
        "max_abs_mc_v": 0.9619,
        "mean_mc": 0.2556,
        "mean_v": 0.1452
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.168,
        "max_abs_mc_v": 0.9047,
        "mean_mc": 0.5725,
        "mean_v": 0.4081
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.004482980817556381,
      "Loss/HJB": 0.04654615372419357,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.03942033648490906,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0044999998062849045,
      "Weights/BDR": 0.0044999998062849045
    }
  },
  {
    "name": "backoff0004_6m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/backoff0004_6m/train/0629_2024_backoff0004_6m_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "backoff0004_6m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.0045,
        0.0045
      ],
      "schedule_final": [
        1.0,
        0.004,
        0.004
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1027,
        "max_abs_mc_v": 0.9671,
        "mean_mc": 0.2497,
        "mean_v": 0.1474
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1625,
        "max_abs_mc_v": 0.9012,
        "mean_mc": 0.5656,
        "mean_v": 0.4069
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10000000149011612,
      "Loss/RL": 0.008495324291288853,
      "Loss/HJB": 0.007141839247196913,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.06562788039445877,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004000000189989805,
      "Weights/BDR": 0.004000000189989805
    }
  },
  {
    "name": "backoff0004_7m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/hold00045_6m/train/0629_2024_hold00045_6m_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_006/backoff0004_7m/train/0630_1311_backoff0004_7m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "backoff0004_7m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/hold00045_6m/train/0629_2024_hold00045_6m_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.0045,
        0.0045
      ],
      "schedule_final": [
        1.0,
        0.004,
        0.004
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.118,
        "max_abs_mc_v": 0.9823,
        "mean_mc": 0.2552,
        "mean_v": 0.1375
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1981,
        "max_abs_mc_v": 0.942,
        "mean_mc": 0.5632,
        "mean_v": 0.3682
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.008263866417109966,
      "Loss/HJB": 0.026816807687282562,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.11698006838560104,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004000000189989805,
      "Weights/BDR": 0.004000000189989805
    }
  },
  {
    "name": "backoff00035_7m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/hold00045_6m/train/0629_2024_hold00045_6m_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_006/backoff00035_7m/train/0630_1311_backoff00035_7m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "backoff00035_7m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/hold00045_6m/train/0629_2024_hold00045_6m_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.0045,
        0.0045
      ],
      "schedule_final": [
        1.0,
        0.0035,
        0.0035
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1085,
        "max_abs_mc_v": 0.9586,
        "mean_mc": 0.2531,
        "mean_v": 0.145
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1751,
        "max_abs_mc_v": 0.9294,
        "mean_mc": 0.5608,
        "mean_v": 0.389
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.07999999821186066,
      "Loss/RL": 0.0021640448831021786,
      "Loss/HJB": 0.024316590279340744,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.0358898788690567,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0035000001080334187,
      "Weights/BDR": 0.0035000001080334187
    }
  },
  {
    "name": "backoff0003_7m",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/hold00045_6m/train/0629_2024_hold00045_6m_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_007/backoff0003_7m/train/0701_0546_backoff0003_7m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "backoff0003_7m",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_005/hold00045_6m/train/0629_2024_hold00045_6m_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.0045,
        0.0045
      ],
      "schedule_final": [
        1.0,
        0.003,
        0.003
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1095,
        "max_abs_mc_v": 0.9553,
        "mean_mc": 0.2525,
        "mean_v": 0.1432
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1856,
        "max_abs_mc_v": 0.9534,
        "mean_mc": 0.5637,
        "mean_v": 0.3814
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.05999999865889549,
      "Loss/RL": 0.010361851193010807,
      "Loss/HJB": 0.0039498815312981606,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.075564444065094,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.003000000026077032,
      "Weights/BDR": 0.003000000026077032
    }
  },
  {
    "name": "earlybackoff00035",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_007/earlybackoff00035/train/0701_0546_earlybackoff00035_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "earlybackoff00035",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_002/hold0005_5m/train/0627_1840_hold0005_5m_seed_1/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.005,
        0.005
      ],
      "schedule_final": [
        1.0,
        0.0035,
        0.0035
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0969,
        "max_abs_mc_v": 0.9236,
        "mean_mc": 0.2523,
        "mean_v": 0.1561
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1594,
        "max_abs_mc_v": 0.9315,
        "mean_mc": 0.5714,
        "mean_v": 0.4156
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.006161837372928858,
      "Loss/HJB": 0.04782724380493164,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.05578359216451645,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0035000001080334187,
      "Weights/BDR": 0.0035000001080334187
    }
  }
]
