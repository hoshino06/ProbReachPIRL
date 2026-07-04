You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/ckpt-2000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_002_plan.json
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
- This is a randT_replayHJB scheduling study from the early 2M randT TD3 checkpoint.
- Treat 10M total updates as the first milestone, not a hard stop; if training remains healthy, continuing toward 12M is acceptable.
- Keep RL weight at 1.0 and compare final reward stability against the TD3 baseline.
- HJB/BDR=0.05 can be tried only after reward and meanMC look stable at smaller weights; 0.05 caused collapse in the 5M-start sweep.
- If average reward decreases monotonically, consider restarting from a safer TD3 or scheduling checkpoint.
- Restart decisions are ambiguous when the degradation is mild. In borderline cases, keep two branches: one continuation branch that tries to recover by extending the current safe checkpoint, and one restart/control branch from a safer checkpoint with reduced or slower HJB/BDR weights.

Reference paths:
- logs/drift/README.md
- logs/drift/scheduling_randT_replayHJB
- logs/drift/td3_T01_randT/
- scheduling_experiment/randT_replayHJB_5Mto10M


Completed results JSON:
[
  {
    "name": "ramp0to001",
    "start_checkpoint": "logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/ckpt-2000000",
    "start_itr": 2000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_000/ramp0to001/train/0626_2257_ramp0to001_seed_1/ckpt-3000000",
    "target_updates": 3000000,
    "candidate": {
      "name": "ramp0to001",
      "start_checkpoint": "logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/ckpt-2000000",
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
        "mean_abs_mc_v": 0.0638,
        "max_abs_mc_v": 0.9032,
        "mean_mc": 0.1803,
        "mean_v": 0.1198
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1442,
        "max_abs_mc_v": 0.7815,
        "mean_mc": 0.5379,
        "mean_v": 0.4228
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.11999999731779099,
      "Loss/RL": 0.004958541132509708,
      "Loss/HJB": 0.20167818665504456,
      "Loss/HJB_replay": 0.06839033216238022,
      "Loss/BDR": 0.06013015657663345,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp0to0005",
    "start_checkpoint": "logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/ckpt-2000000",
    "start_itr": 2000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_000/ramp0to0005/train/0626_2257_ramp0to0005_seed_1/ckpt-3000000",
    "target_updates": 3000000,
    "candidate": {
      "name": "ramp0to0005",
      "start_checkpoint": "logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/ckpt-2000000",
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
        "mean_abs_mc_v": 0.105,
        "max_abs_mc_v": 0.9788,
        "mean_mc": 0.2683,
        "mean_v": 0.1684
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1233,
        "max_abs_mc_v": 0.8247,
        "mean_mc": 0.5888,
        "mean_v": 0.4815
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.07999999821186066,
      "Loss/RL": 0.003417012747377157,
      "Loss/HJB": 0.23778808116912842,
      "Loss/HJB_replay": 0.1307564228773117,
      "Loss/BDR": 0.0877586379647255,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "const001_3m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_000/ramp0to001/train/0626_2257_ramp0to001_seed_1/ckpt-3000000",
    "start_itr": 3000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_001/const001_3m/train/0627_1313_const001_3m_seed_1/ckpt-4000000",
    "target_updates": 4000000,
    "candidate": {
      "name": "const001_3m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_000/ramp0to001/train/0626_2257_ramp0to001_seed_1/ckpt-3000000",
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
        "mean_abs_mc_v": 0.0888,
        "max_abs_mc_v": 0.9585,
        "mean_mc": 0.2395,
        "mean_v": 0.1527
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1147,
        "max_abs_mc_v": 0.8253,
        "mean_mc": 0.5895,
        "mean_v": 0.488
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.0042577702552080154,
      "Loss/HJB": 0.2110690325498581,
      "Loss/HJB_replay": 0.08245709538459778,
      "Loss/BDR": 0.036080971360206604,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "const0005_3m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_000/ramp0to0005/train/0626_2257_ramp0to0005_seed_1/ckpt-3000000",
    "start_itr": 3000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_001/const0005_3m/train/0627_1313_const0005_3m_seed_1/ckpt-4000000",
    "target_updates": 4000000,
    "candidate": {
      "name": "const0005_3m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_000/ramp0to0005/train/0626_2257_ramp0to0005_seed_1/ckpt-3000000",
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
        "mean_abs_mc_v": 0.0893,
        "max_abs_mc_v": 0.9339,
        "mean_mc": 0.2639,
        "mean_v": 0.1768
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1033,
        "max_abs_mc_v": 0.8141,
        "mean_mc": 0.5911,
        "mean_v": 0.4995
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.003514673560857773,
      "Loss/HJB": 0.2980283498764038,
      "Loss/HJB_replay": 0.11258502304553986,
      "Loss/BDR": 0.028963223099708557,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "const0005_4m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_001/const0005_3m/train/0627_1313_const0005_3m_seed_1/ckpt-4000000",
    "start_itr": 4000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_002/const0005_4m/train/0628_0307_const0005_4m_seed_1/ckpt-5000000",
    "target_updates": 5000000,
    "candidate": {
      "name": "const0005_4m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_001/const0005_3m/train/0627_1313_const0005_3m_seed_1/ckpt-4000000",
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
        "mean_abs_mc_v": 0.0893,
        "max_abs_mc_v": 0.9578,
        "mean_mc": 0.2702,
        "mean_v": 0.1824
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.107,
        "max_abs_mc_v": 0.8751,
        "mean_mc": 0.5961,
        "mean_v": 0.4984
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.001682340749539435,
      "Loss/HJB": 0.4270496964454651,
      "Loss/HJB_replay": 0.12906408309936523,
      "Loss/BDR": 0.0657278448343277,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "ramp001to0015_4m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_001/const001_3m/train/0627_1313_const001_3m_seed_1/ckpt-4000000",
    "start_itr": 4000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_002/ramp001to0015_4m/train/0628_0307_ramp001to0015_4m_seed_1/ckpt-5000000",
    "target_updates": 5000000,
    "candidate": {
      "name": "ramp001to0015_4m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_001/const001_3m/train/0627_1313_const001_3m_seed_1/ckpt-4000000",
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
        "mean_abs_mc_v": 0.0727,
        "max_abs_mc_v": 0.9188,
        "mean_mc": 0.2142,
        "mean_v": 0.1464
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1097,
        "max_abs_mc_v": 0.8778,
        "mean_mc": 0.5818,
        "mean_v": 0.4824
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.004276202991604805,
      "Loss/HJB": 0.20003226399421692,
      "Loss/HJB_replay": 0.06510041654109955,
      "Loss/BDR": 0.033234190195798874,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.014999999664723873,
      "Weights/BDR": 0.014999999664723873
    }
  }
]
