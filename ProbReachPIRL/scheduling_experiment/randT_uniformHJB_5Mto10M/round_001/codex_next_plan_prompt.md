You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/user/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/randT_uniformHJB_5Mto10M/round_002_plan.json
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
  }
]
