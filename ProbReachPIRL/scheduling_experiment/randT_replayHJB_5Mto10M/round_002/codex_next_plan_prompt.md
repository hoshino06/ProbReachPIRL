You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_003_plan.json
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
- This matches manual randT_replayHJB: randT TD3 baseline, randT scheduling reset, replay-memory HJB samples.
- Keep RL weight at 1.0 and compare reward stability against the randT TD3 baseline.
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
    "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_000/ramp0to001/train/0625_1832_ramp0to001_seed_1/ckpt-6000000",
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
        "mean_abs_mc_v": 0.0971,
        "max_abs_mc_v": 0.9957,
        "mean_mc": 0.233,
        "mean_v": 0.1395
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1309,
        "max_abs_mc_v": 0.7795,
        "mean_mc": 0.5628,
        "mean_v": 0.4394
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10999999940395355,
      "Loss/RL": 0.003545330371707678,
      "Loss/HJB": 0.15828561782836914,
      "Loss/HJB_replay": 0.08322634547948837,
      "Loss/BDR": 0.04875287786126137,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp0to005",
    "start_checkpoint": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_000/ramp0to005/train/0625_1832_ramp0to005_seed_1/ckpt-6000000",
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
        "mean_abs_mc_v": 0.0131,
        "max_abs_mc_v": 0.9939,
        "mean_mc": 0.0212,
        "mean_v": 0.0105
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0015,
        "max_abs_mc_v": 0.3613,
        "mean_mc": 0.0137,
        "mean_v": 0.0149
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.009999999776482582,
      "Loss/RL": 2.5272855054936372e-05,
      "Loss/HJB": 0.020084798336029053,
      "Loss/HJB_replay": 0.0030010028276592493,
      "Loss/BDR": 0.0018014368833974004,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.05000000074505806,
      "Weights/BDR": 0.05000000074505806
    }
  },
  {
    "name": "const001_6m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_000/ramp0to001/train/0625_1832_ramp0to001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_001/const001_6m/train/0626_1255_const001_6m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "const001_6m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_000/ramp0to001/train/0625_1832_ramp0to001_seed_1/ckpt-6000000",
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
        "mean_abs_mc_v": 0.0821,
        "max_abs_mc_v": 0.967,
        "mean_mc": 0.2224,
        "mean_v": 0.1428
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1048,
        "max_abs_mc_v": 0.7598,
        "mean_mc": 0.5479,
        "mean_v": 0.4573
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.002504831412807107,
      "Loss/HJB": 0.15083862841129303,
      "Loss/HJB_replay": 0.10221925377845764,
      "Loss/BDR": 0.02826811373233795,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp001to002",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_000/ramp0to001/train/0625_1832_ramp0to001_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_001/ramp001to002/train/0626_1255_ramp001to002_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "ramp001to002",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_000/ramp0to001/train/0625_1832_ramp0to001_seed_1/ckpt-6000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
      ],
      "schedule_final": [
        1.0,
        0.02,
        0.02
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0653,
        "max_abs_mc_v": 0.9162,
        "mean_mc": 0.1755,
        "mean_v": 0.1259
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1275,
        "max_abs_mc_v": 0.7892,
        "mean_mc": 0.5145,
        "mean_v": 0.4154
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.09000000357627869,
      "Loss/RL": 0.005772299598902464,
      "Loss/HJB": 0.2326871156692505,
      "Loss/HJB_replay": 0.048440225422382355,
      "Loss/BDR": 0.057732220739126205,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.019999999552965164,
      "Weights/BDR": 0.019999999552965164
    }
  },
  {
    "name": "const001_7m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_001/const001_6m/train/0626_1255_const001_6m_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_002/const001_7m/train/0627_0304_const001_7m_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "const001_7m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_001/const001_6m/train/0626_1255_const001_6m_seed_1/ckpt-7000000",
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
        "mean_abs_mc_v": 0.0917,
        "max_abs_mc_v": 0.9549,
        "mean_mc": 0.2417,
        "mean_v": 0.1514
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1171,
        "max_abs_mc_v": 0.8828,
        "mean_mc": 0.5744,
        "mean_v": 0.4631
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.07000000029802322,
      "Loss/RL": 0.004216436296701431,
      "Loss/HJB": 0.15113694965839386,
      "Loss/HJB_replay": 0.0504155270755291,
      "Loss/BDR": 0.024315226823091507,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp001to0015",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_001/const001_6m/train/0626_1255_const001_6m_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_002/ramp001to0015/train/0627_0304_ramp001to0015_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "ramp001to0015",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_5Mto10M/round_001/const001_6m/train/0626_1255_const001_6m_seed_1/ckpt-7000000",
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
        "mean_abs_mc_v": 0.0724,
        "max_abs_mc_v": 0.9295,
        "mean_mc": 0.2092,
        "mean_v": 0.139
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1233,
        "max_abs_mc_v": 0.8736,
        "mean_mc": 0.5645,
        "mean_v": 0.4475
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.0015980741009116173,
      "Loss/HJB": 0.04488424211740494,
      "Loss/HJB_replay": 0.050284817814826965,
      "Loss/BDR": 0.016724741086363792,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.014999999664723873,
      "Weights/BDR": 0.014999999664723873
    }
  }
]
