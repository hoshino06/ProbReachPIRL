You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_008_plan.json
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
- This is the randT counterpart to fixed2randT_uniformHJB: randT TD3 baseline, randT scheduling reset, uniform HJB samples.
- Compare against randT_replayHJB_5Mto10M to isolate the HJB sampling distribution.
- Compare against fixed2randT_uniformHJB_5Mto10M to isolate whether the higher-reward fixedT baseline transfers better under randT scheduling.
- Keep RL weight at 1.0. The goal is to reduce mean|MC-V| without making final reward worse than the randT TD3 baseline.
- Track both reward stability and MC/value calibration, but do not restart only because one intermediate round is ambiguous.

Reference paths:
- logs/drift/README.md
- logs/drift/scheduling_randT_replayHJB
- logs/drift/scheduling_fixed2randT_uniformHJB
- logs/drift/scheduling_fixedT_replayHJB
- logs/drift/scheduling_fixedT_uniformHJB
- scheduling_experiment/randT_replayHJB_5Mto10M
- scheduling_experiment/fixed2randT_uniformHJB_5Mto10M
- logs/drift/td3_T01_randT/
- logs/drift/td3_T01/


Completed results JSON:
[
  {
    "name": "ramp0to001",
    "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_000/ramp0to001/train/0630_1110_ramp0to001_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "ramp0to001",
      "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
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
        "mean_abs_mc_v": 0.1019,
        "max_abs_mc_v": 0.9926,
        "mean_mc": 0.2641,
        "mean_v": 0.1628
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1611,
        "max_abs_mc_v": 0.8883,
        "mean_mc": 0.5905,
        "mean_v": 0.4344
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1899999976158142,
      "Loss/RL": 0.005122005473822355,
      "Loss/HJB": 0.0373312272131443,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.04115765914320946,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp0to005",
    "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_000/ramp0to005/train/0630_1110_ramp0to005_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "ramp0to005",
      "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
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
        "mean_abs_mc_v": 0.1205,
        "max_abs_mc_v": 0.9844,
        "mean_mc": 0.2577,
        "mean_v": 0.1376
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1799,
        "max_abs_mc_v": 0.8744,
        "mean_mc": 0.5691,
        "mean_v": 0.3943
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.17000000178813934,
      "Loss/RL": 0.008566390722990036,
      "Loss/HJB": 0.006752415560185909,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.005418634042143822,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.05000000074505806,
      "Weights/BDR": 0.05000000074505806
    }
  },
  {
    "name": "const001_6m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_000/ramp0to001/train/0630_1110_ramp0to001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_001/const001_6m/train/0701_0015_const001_6m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "const001_6m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_000/ramp0to001/train/0630_1110_ramp0to001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
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
        "mean_abs_mc_v": 0.102,
        "max_abs_mc_v": 0.9802,
        "mean_mc": 0.2607,
        "mean_v": 0.1604
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1796,
        "max_abs_mc_v": 0.898,
        "mean_mc": 0.5825,
        "mean_v": 0.4073
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10000000149011612,
      "Loss/RL": 0.01114528439939022,
      "Loss/HJB": 0.01666376367211342,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.0271643977612257,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp001to0015",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_000/ramp0to001/train/0630_1110_ramp0to001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_001/ramp001to0015/train/0701_0015_ramp001to0015_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "ramp001to0015",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_000/ramp0to001/train/0630_1110_ramp0to001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
      ],
      "schedule_final": [
        1.0,
        0.015,
        0.015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0998,
        "max_abs_mc_v": 0.9887,
        "mean_mc": 0.2601,
        "mean_v": 0.1615
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1672,
        "max_abs_mc_v": 0.8385,
        "mean_mc": 0.5831,
        "mean_v": 0.4216
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.0032076435163617134,
      "Loss/HJB": 0.026219189167022705,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.034910622984170914,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.014999999664723873,
      "Weights/BDR": 0.014999999664723873
    }
  },
  {
    "name": "const0015_7m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_001/ramp001to0015/train/0701_0015_ramp001to0015_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_002/const0015_7m/train/0701_1346_const0015_7m_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "const0015_7m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_001/ramp001to0015/train/0701_0015_ramp001to0015_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.015,
        0.015
      ],
      "schedule_final": [
        1.0,
        0.015,
        0.015
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1209,
        "max_abs_mc_v": 0.9893,
        "mean_mc": 0.258,
        "mean_v": 0.1377
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2111,
        "max_abs_mc_v": 0.944,
        "mean_mc": 0.5622,
        "mean_v": 0.3546
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10999999940395355,
      "Loss/RL": 0.0055604432709515095,
      "Loss/HJB": 0.0146494684740901,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.01752510480582714,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.014999999664723873,
      "Weights/BDR": 0.014999999664723873
    }
  },
  {
    "name": "taper0012_7m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_001/ramp001to0015/train/0701_0015_ramp001to0015_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_002/taper0012_7m/train/0701_1346_taper0012_7m_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "taper0012_7m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_001/ramp001to0015/train/0701_0015_ramp001to0015_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.015,
        0.015
      ],
      "schedule_final": [
        1.0,
        0.012,
        0.012
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1204,
        "max_abs_mc_v": 0.9881,
        "mean_mc": 0.259,
        "mean_v": 0.1392
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2166,
        "max_abs_mc_v": 0.9311,
        "mean_mc": 0.5618,
        "mean_v": 0.3477
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.004206774290651083,
      "Loss/HJB": 0.03293542191386223,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.04315991699695587,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.012000000104308128,
      "Weights/BDR": 0.012000000104308128
    }
  },
  {
    "name": "const0012_8m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_002/taper0012_7m/train/0701_1346_taper0012_7m_seed_1/ckpt-8000000",
    "start_itr": 8000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_003/const0012_8m/train/0702_0820_const0012_8m_seed_1/ckpt-9000000",
    "target_updates": 9000000,
    "candidate": {
      "name": "const0012_8m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_002/taper0012_7m/train/0701_1346_taper0012_7m_seed_1/ckpt-8000000",
      "schedule_initial": [
        1.0,
        0.012,
        0.012
      ],
      "schedule_final": [
        1.0,
        0.012,
        0.012
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1352,
        "max_abs_mc_v": 0.9536,
        "mean_mc": 0.2581,
        "mean_v": 0.1238
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2521,
        "max_abs_mc_v": 0.9466,
        "mean_mc": 0.5659,
        "mean_v": 0.3162
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.010004868730902672,
      "Loss/HJB": 0.00610174797475338,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.021001599729061127,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.012000000104308128,
      "Weights/BDR": 0.012000000104308128
    }
  },
  {
    "name": "taper0010_8m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_002/taper0012_7m/train/0701_1346_taper0012_7m_seed_1/ckpt-8000000",
    "start_itr": 8000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_003/taper0010_8m/train/0702_0820_taper0010_8m_seed_1/ckpt-9000000",
    "target_updates": 9000000,
    "candidate": {
      "name": "taper0010_8m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_002/taper0012_7m/train/0701_1346_taper0012_7m_seed_1/ckpt-8000000",
      "schedule_initial": [
        1.0,
        0.012,
        0.012
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
        "mean_abs_mc_v": 0.1285,
        "max_abs_mc_v": 0.9649,
        "mean_mc": 0.2525,
        "mean_v": 0.1251
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2447,
        "max_abs_mc_v": 0.9004,
        "mean_mc": 0.5619,
        "mean_v": 0.3195
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.11999999731779099,
      "Loss/RL": 0.016447369009256363,
      "Loss/HJB": 0.0358419306576252,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.026679735630750656,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "taper0010_9m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_003/const0012_8m/train/0702_0820_const0012_8m_seed_1/ckpt-9000000",
    "start_itr": 9000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_004/taper0010_9m/train/0703_0420_taper0010_9m_seed_1/ckpt-10000000",
    "target_updates": 10000000,
    "candidate": {
      "name": "taper0010_9m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_003/const0012_8m/train/0702_0820_const0012_8m_seed_1/ckpt-9000000",
      "schedule_initial": [
        1.0,
        0.012,
        0.012
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
        "mean_abs_mc_v": 0.1329,
        "max_abs_mc_v": 0.9772,
        "mean_mc": 0.2524,
        "mean_v": 0.1208
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2572,
        "max_abs_mc_v": 0.9425,
        "mean_mc": 0.5576,
        "mean_v": 0.3031
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.09000000357627869,
      "Loss/RL": 0.004223903641104698,
      "Loss/HJB": 0.8987259864807129,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.0306355282664299,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "const0010_9m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_003/taper0010_8m/train/0702_0820_taper0010_8m_seed_1/ckpt-9000000",
    "start_itr": 9000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_004/const0010_9m/train/0703_0420_const0010_9m_seed_1/ckpt-10000000",
    "target_updates": 10000000,
    "candidate": {
      "name": "const0010_9m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_003/taper0010_8m/train/0702_0820_taper0010_8m_seed_1/ckpt-9000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
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
        "mean_abs_mc_v": 0.136,
        "max_abs_mc_v": 0.9603,
        "mean_mc": 0.2552,
        "mean_v": 0.1204
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2591,
        "max_abs_mc_v": 0.9219,
        "mean_mc": 0.5624,
        "mean_v": 0.306
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.11999999731779099,
      "Loss/RL": 0.006057461723685265,
      "Loss/HJB": 0.017105478793382645,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.028976595029234886,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "const0010_10m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_004/const0010_9m/train/0703_0420_const0010_9m_seed_1/ckpt-10000000",
    "start_itr": 10000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/const0010_10m/train/0704_0018_const0010_10m_seed_1/ckpt-11000000",
    "target_updates": 11000000,
    "candidate": {
      "name": "const0010_10m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_004/const0010_9m/train/0703_0420_const0010_9m_seed_1/ckpt-10000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
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
        "mean_abs_mc_v": 0.1357,
        "max_abs_mc_v": 0.9421,
        "mean_mc": 0.2544,
        "mean_v": 0.1194
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2685,
        "max_abs_mc_v": 0.9418,
        "mean_mc": 0.5616,
        "mean_v": 0.2955
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10000000149011612,
      "Loss/RL": 0.008925278671085835,
      "Loss/HJB": 0.02776782028377056,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.017564887180924416,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "taper0008_10m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_004/const0010_9m/train/0703_0420_const0010_9m_seed_1/ckpt-10000000",
    "start_itr": 10000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/taper0008_10m/train/0704_0018_taper0008_10m_seed_1/ckpt-11000000",
    "target_updates": 11000000,
    "candidate": {
      "name": "taper0008_10m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_004/const0010_9m/train/0703_0420_const0010_9m_seed_1/ckpt-10000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
      ],
      "schedule_final": [
        1.0,
        0.008,
        0.008
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1402,
        "max_abs_mc_v": 0.9575,
        "mean_mc": 0.2502,
        "mean_v": 0.1107
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2883,
        "max_abs_mc_v": 0.9423,
        "mean_mc": 0.5635,
        "mean_v": 0.2773
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.010835371911525726,
      "Loss/HJB": 0.009022406302392483,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.03387929126620293,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.00800000037997961,
      "Weights/BDR": 0.00800000037997961
    }
  },
  {
    "name": "taper0008_11m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/const0010_10m/train/0704_0018_const0010_10m_seed_1/ckpt-11000000",
    "start_itr": 11000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_006/taper0008_11m/train/0704_1634_taper0008_11m_seed_1/ckpt-12000000",
    "target_updates": 12000000,
    "candidate": {
      "name": "taper0008_11m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/const0010_10m/train/0704_0018_const0010_10m_seed_1/ckpt-11000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
      ],
      "schedule_final": [
        1.0,
        0.008,
        0.008
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1458,
        "max_abs_mc_v": 0.9704,
        "mean_mc": 0.26,
        "mean_v": 0.1146
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2817,
        "max_abs_mc_v": 0.9403,
        "mean_mc": 0.5526,
        "mean_v": 0.2725
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.09000000357627869,
      "Loss/RL": 0.005017315037548542,
      "Loss/HJB": 0.0016330545768141747,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.032515138387680054,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.00800000037997961,
      "Weights/BDR": 0.00800000037997961
    }
  },
  {
    "name": "taper0006_11m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/const0010_10m/train/0704_0018_const0010_10m_seed_1/ckpt-11000000",
    "start_itr": 11000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_006/taper0006_11m/train/0704_1634_taper0006_11m_seed_1/ckpt-12000000",
    "target_updates": 12000000,
    "candidate": {
      "name": "taper0006_11m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/const0010_10m/train/0704_0018_const0010_10m_seed_1/ckpt-11000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
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
        "mean_abs_mc_v": 0.1524,
        "max_abs_mc_v": 0.9671,
        "mean_mc": 0.2562,
        "mean_v": 0.1044
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2898,
        "max_abs_mc_v": 0.9335,
        "mean_mc": 0.544,
        "mean_v": 0.2561
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10999999940395355,
      "Loss/RL": 0.00640293350443244,
      "Loss/HJB": 0.012871455401182175,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.023102417588233948,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.006000000052154064,
      "Weights/BDR": 0.006000000052154064
    }
  },
  {
    "name": "hold0008_11m_alt",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/taper0008_10m/train/0704_0018_taper0008_10m_seed_1/ckpt-11000000",
    "start_itr": 11000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_007/hold0008_11m_alt/train/0705_0615_hold0008_11m_alt_seed_1/ckpt-12000000",
    "target_updates": 12000000,
    "candidate": {
      "name": "hold0008_11m_alt",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_005/taper0008_10m/train/0704_0018_taper0008_10m_seed_1/ckpt-11000000",
      "schedule_initial": [
        1.0,
        0.008,
        0.008
      ],
      "schedule_final": [
        1.0,
        0.008,
        0.008
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1452,
        "max_abs_mc_v": 0.9455,
        "mean_mc": 0.2576,
        "mean_v": 0.1131
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.2883,
        "max_abs_mc_v": 0.9292,
        "mean_mc": 0.551,
        "mean_v": 0.2649
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.14000000059604645,
      "Loss/RL": 0.009043100290000439,
      "Loss/HJB": 0.0660250335931778,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.026393236592411995,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.00800000037997961,
      "Weights/BDR": 0.00800000037997961
    }
  },
  {
    "name": "restart_ramp0005",
    "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_007/restart_ramp0005/train/0705_0615_restart_ramp0005_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "restart_ramp0005",
      "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
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
        "mean_abs_mc_v": 0.0939,
        "max_abs_mc_v": 0.9361,
        "mean_mc": 0.2644,
        "mean_v": 0.1713
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1572,
        "max_abs_mc_v": 0.859,
        "mean_mc": 0.5965,
        "mean_v": 0.4449
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.11999999731779099,
      "Loss/RL": 0.003344150260090828,
      "Loss/HJB": 0.012435509823262691,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.054369207471609116,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  }
]
