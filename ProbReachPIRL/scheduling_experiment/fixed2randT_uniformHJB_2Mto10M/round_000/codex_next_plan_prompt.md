You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01/up03M_scale10_mix334/ckpt-2000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_2Mto10M/round_001_plan.json
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
  }
]
