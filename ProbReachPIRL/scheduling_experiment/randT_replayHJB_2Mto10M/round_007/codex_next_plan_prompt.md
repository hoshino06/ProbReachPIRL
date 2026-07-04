You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01_randT/up02M_scale08_mix334_randT/ckpt-2000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_008_plan.json
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
  },
  {
    "name": "const0005_5m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_002/const0005_4m/train/0628_0307_const0005_4m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_003/const0005_5m/train/0629_1004_const0005_5m_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "const0005_5m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_002/const0005_4m/train/0628_0307_const0005_4m_seed_1/ckpt-5000000",
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
        "mean_abs_mc_v": 0.084,
        "max_abs_mc_v": 0.9711,
        "mean_mc": 0.267,
        "mean_v": 0.1841
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1076,
        "max_abs_mc_v": 0.8038,
        "mean_mc": 0.5923,
        "mean_v": 0.4925
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.12999999523162842,
      "Loss/RL": 0.004861100576817989,
      "Loss/HJB": 0.4611877501010895,
      "Loss/HJB_replay": 0.1134953498840332,
      "Loss/BDR": 0.07398750633001328,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "recover001_5m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_002/ramp001to0015_4m/train/0628_0307_ramp001to0015_4m_seed_1/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_003/recover001_5m/train/0629_1004_recover001_5m_seed_1/ckpt-6000000",
    "target_updates": 6000000,
    "candidate": {
      "name": "recover001_5m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_002/ramp001to0015_4m/train/0628_0307_ramp001to0015_4m_seed_1/ckpt-5000000",
      "schedule_initial": [
        1.0,
        0.015,
        0.015
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
        "mean_abs_mc_v": 0.085,
        "max_abs_mc_v": 0.9171,
        "mean_mc": 0.2445,
        "mean_v": 0.1619
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1087,
        "max_abs_mc_v": 0.7702,
        "mean_mc": 0.5909,
        "mean_v": 0.4928
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.0036869700998067856,
      "Loss/HJB": 0.195512592792511,
      "Loss/HJB_replay": 0.08181820809841156,
      "Loss/BDR": 0.005389691796153784,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "const0005_6m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_003/const0005_5m/train/0629_1004_const0005_5m_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_004/const0005_6m/train/0629_2316_const0005_6m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "const0005_6m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_003/const0005_5m/train/0629_1004_const0005_5m_seed_1/ckpt-6000000",
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
        "mean_abs_mc_v": 0.0811,
        "max_abs_mc_v": 0.9425,
        "mean_mc": 0.2645,
        "mean_v": 0.1847
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1035,
        "max_abs_mc_v": 0.8276,
        "mean_mc": 0.5901,
        "mean_v": 0.4946
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10999999940395355,
      "Loss/RL": 0.01033850573003292,
      "Loss/HJB": 0.19554896652698517,
      "Loss/HJB_replay": 0.10080491006374359,
      "Loss/BDR": 0.03668110445141792,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.004999999888241291,
      "Weights/BDR": 0.004999999888241291
    }
  },
  {
    "name": "const001_6m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_003/recover001_5m/train/0629_1004_recover001_5m_seed_1/ckpt-6000000",
    "start_itr": 6000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_004/const001_6m/train/0629_2316_const001_6m_seed_1/ckpt-7000000",
    "target_updates": 7000000,
    "candidate": {
      "name": "const001_6m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_003/recover001_5m/train/0629_1004_recover001_5m_seed_1/ckpt-6000000",
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
        "mean_abs_mc_v": 0.0902,
        "max_abs_mc_v": 0.8804,
        "mean_mc": 0.2499,
        "mean_v": 0.1613
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1058,
        "max_abs_mc_v": 0.8841,
        "mean_mc": 0.5888,
        "mean_v": 0.4911
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.003331131301820278,
      "Loss/HJB": 0.2794523537158966,
      "Loss/HJB_replay": 0.05119699984788895,
      "Loss/BDR": 0.007340774405747652,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "const001_7m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_004/const001_6m/train/0629_2316_const001_6m_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_005/const001_7m/train/0630_1248_const001_7m_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "const001_7m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_004/const001_6m/train/0629_2316_const001_6m_seed_1/ckpt-7000000",
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
        "mean_abs_mc_v": 0.0905,
        "max_abs_mc_v": 0.9308,
        "mean_mc": 0.2479,
        "mean_v": 0.1636
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0998,
        "max_abs_mc_v": 0.8269,
        "mean_mc": 0.572,
        "mean_v": 0.4857
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.0021322397515177727,
      "Loss/HJB": 0.40705573558807373,
      "Loss/HJB_replay": 0.06791224330663681,
      "Loss/BDR": 0.02133072167634964,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp001to00125_7m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_004/const001_6m/train/0629_2316_const001_6m_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_005/ramp001to00125_7m/train/0630_1248_ramp001to00125_7m_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "ramp001to00125_7m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_004/const001_6m/train/0629_2316_const001_6m_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
      ],
      "schedule_final": [
        1.0,
        0.0125,
        0.0125
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0838,
        "max_abs_mc_v": 0.8371,
        "mean_mc": 0.2402,
        "mean_v": 0.161
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0913,
        "max_abs_mc_v": 0.8321,
        "mean_mc": 0.559,
        "mean_v": 0.485
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10000000149011612,
      "Loss/RL": 0.003475356847047806,
      "Loss/HJB": 0.3079516887664795,
      "Loss/HJB_replay": 0.0418291836977005,
      "Loss/BDR": 0.027010073885321617,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.012500000186264515,
      "Weights/BDR": 0.012500000186264515
    }
  },
  {
    "name": "const001_8m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_005/const001_7m/train/0630_1248_const001_7m_seed_1/ckpt-8000000",
    "start_itr": 8000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_006/const001_8m/train/0701_0215_const001_8m_seed_1/ckpt-9000000",
    "target_updates": 9000000,
    "candidate": {
      "name": "const001_8m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_005/const001_7m/train/0630_1248_const001_7m_seed_1/ckpt-8000000",
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
        "mean_abs_mc_v": 0.093,
        "max_abs_mc_v": 0.8976,
        "mean_mc": 0.26,
        "mean_v": 0.1696
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1053,
        "max_abs_mc_v": 0.8911,
        "mean_mc": 0.5876,
        "mean_v": 0.4892
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.11999999731779099,
      "Loss/RL": 0.004500973038375378,
      "Loss/HJB": 0.3906291723251343,
      "Loss/HJB_replay": 0.0527559369802475,
      "Loss/BDR": 0.012340601533651352,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "slow001to0011_8m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_005/const001_7m/train/0630_1248_const001_7m_seed_1/ckpt-8000000",
    "start_itr": 8000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_006/slow001to0011_8m/train/0701_0215_slow001to0011_8m_seed_1/ckpt-9000000",
    "target_updates": 9000000,
    "candidate": {
      "name": "slow001to0011_8m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_005/const001_7m/train/0630_1248_const001_7m_seed_1/ckpt-8000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
      ],
      "schedule_final": [
        1.0,
        0.011,
        0.011
      ],
      "schedule_center": 750000,
      "schedule_sharpness": 5e-06
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.0851,
        "max_abs_mc_v": 0.8323,
        "mean_mc": 0.2484,
        "mean_v": 0.1662
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1011,
        "max_abs_mc_v": 0.8092,
        "mean_mc": 0.5791,
        "mean_v": 0.4878
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.05999999865889549,
      "Loss/RL": 0.005859797354787588,
      "Loss/HJB": 0.1968512088060379,
      "Loss/HJB_replay": 0.04820388928055763,
      "Loss/BDR": 0.03752467781305313,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.010777299292385578,
      "Weights/BDR": 0.010777299292385578
    }
  },
  {
    "name": "recover00075_9m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_006/const001_8m/train/0701_0215_const001_8m_seed_1/ckpt-9000000",
    "start_itr": 9000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_007/recover00075_9m/train/0701_1536_recover00075_9m_seed_1/ckpt-10000000",
    "target_updates": 10000000,
    "candidate": {
      "name": "recover00075_9m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_006/const001_8m/train/0701_0215_const001_8m_seed_1/ckpt-9000000",
      "schedule_initial": [
        1.0,
        0.01,
        0.01
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
        "mean_abs_mc_v": 0.0825,
        "max_abs_mc_v": 0.8445,
        "mean_mc": 0.2558,
        "mean_v": 0.177
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0922,
        "max_abs_mc_v": 0.8981,
        "mean_mc": 0.5749,
        "mean_v": 0.501
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.007939890958368778,
      "Loss/HJB": 0.163255512714386,
      "Loss/HJB_replay": 0.07839858531951904,
      "Loss/BDR": 0.008013227954506874,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.007499999832361937,
      "Weights/BDR": 0.007499999832361937
    }
  },
  {
    "name": "slowrecover0008_9m",
    "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_006/slow001to0011_8m/train/0701_0215_slow001to0011_8m_seed_1/ckpt-9000000",
    "start_itr": 9000000,
    "checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_007/slowrecover0008_9m/train/0701_1536_slowrecover0008_9m_seed_1/ckpt-10000000",
    "target_updates": 10000000,
    "candidate": {
      "name": "slowrecover0008_9m",
      "start_checkpoint": "/home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_replayHJB_2Mto10M/round_006/slow001to0011_8m/train/0701_0215_slow001to0011_8m_seed_1/ckpt-9000000",
      "schedule_initial": [
        1.0,
        0.011,
        0.011
      ],
      "schedule_final": [
        1.0,
        0.008,
        0.008
      ],
      "schedule_center": 750000,
      "schedule_sharpness": 5e-06
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.091,
        "max_abs_mc_v": 0.8261,
        "mean_mc": 0.2586,
        "mean_v": 0.17
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0995,
        "max_abs_mc_v": 0.867,
        "mean_mc": 0.5844,
        "mean_v": 0.492
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.15000000596046448,
      "Loss/RL": 0.005698409862816334,
      "Loss/HJB": 0.4479556083679199,
      "Loss/HJB_replay": 0.048631638288497925,
      "Loss/BDR": 0.03158583119511604,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.008668103255331516,
      "Weights/BDR": 0.008668103255331516
    }
  }
]
