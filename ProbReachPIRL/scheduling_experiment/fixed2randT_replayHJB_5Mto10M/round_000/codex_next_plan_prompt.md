You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M/round_001_plan.json
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
- This tests the fixedT TD3 to randT scheduling switch while using replay-memory HJB samples.
- Compare against fixedT_replayHJB to isolate the reset-T change, and against fixed2randT_uniformHJB to isolate the HJB sampling distribution.
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
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M/round_000/ramp0to001/train/0625_1823_ramp0to001_seed_1/ckpt-6000000",
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
        "mean_abs_mc_v": 0.0074,
        "max_abs_mc_v": 0.7375,
        "mean_mc": 0.0202,
        "mean_v": 0.0234
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0038,
        "max_abs_mc_v": 0.7186,
        "mean_mc": 0.0128,
        "mean_v": 0.0148
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.029999999329447746,
      "Loss/RL": 0.006470320280641317,
      "Loss/HJB": 0.21885274350643158,
      "Loss/HJB_replay": 0.023100387305021286,
      "Loss/BDR": 0.06153598427772522,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.009999999776482582,
      "Weights/BDR": 0.009999999776482582
    }
  },
  {
    "name": "ramp0to005",
    "start_checkpoint": "logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000",
    "start_itr": 5000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_5Mto10M/round_000/ramp0to005/train/0625_1823_ramp0to005_seed_1/ckpt-6000000",
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
        "mean_abs_mc_v": 0.0096,
        "max_abs_mc_v": 0.8952,
        "mean_mc": 0.025,
        "mean_v": 0.0272
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0018,
        "max_abs_mc_v": 0.269,
        "mean_mc": 0.0125,
        "mean_v": 0.0141
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.009999999776482582,
      "Loss/RL": 3.68324872397352e-05,
      "Loss/HJB": 0.025439070537686348,
      "Loss/HJB_replay": 0.013802316971123219,
      "Loss/BDR": 0.006000156048685312,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.05000000074505806,
      "Weights/BDR": 0.05000000074505806
    }
  }
]
