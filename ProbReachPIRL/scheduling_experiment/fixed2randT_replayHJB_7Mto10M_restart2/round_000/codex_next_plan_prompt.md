You are controlling the next round of PIRL weight scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from logs/drift/td3_T01/up05M_scale10_mix334/ckpt-5000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.

Output:
- Write ONLY valid JSON to: /home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_001_plan.json
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
- This restart2 continues from fixed2randT_replayHJB_5Mto10M_restart round_001, because TensorBoard smoothed reward suggested those branches were more promising than the final raw reward alone indicated.
- Reward is noisy. Do not reject a checkpoint based only on the final raw reward value.
- Avoid fixed absolute reward thresholds unless they are justified by the current experiment. Compare candidates relative to nearby runs under the same reset and sampling setup.
- Prefer continuations that keep reward comparable while improving MC calibration. Increase HJB/BDR gradually when the reward trend remains acceptable.

Reference paths:
- logs/drift/README.md
- logs/drift/scheduling_randT_replayHJB
- logs/drift/scheduling_fixed2randT_uniformHJB
- logs/drift/scheduling_fixedT_replayHJB
- logs/drift/scheduling_fixedT_uniformHJB
- logs/drift/td3_T01_randT/
- logs/drift/td3_T01/
- scheduling_experiment/fixed2randT_replayHJB_5Mto10M
- scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart


Completed results JSON:
[
  {
    "name": "hold0001_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_001/hold0001/train/0627_0211_hold0001_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_000/hold0001_R1/train/0630_0945_hold0001_R1_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "hold0001_R1",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_001/hold0001/train/0627_0211_hold0001_seed_1/ckpt-7000000",
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
        "mean_abs_mc_v": 0.1188,
        "max_abs_mc_v": 0.9959,
        "mean_mc": 0.4137,
        "mean_v": 0.2953
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1021,
        "max_abs_mc_v": 0.872,
        "mean_mc": 0.588,
        "mean_v": 0.4916
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.1599999964237213,
      "Loss/RL": 0.008215537294745445,
      "Loss/HJB": 0.7740164399147034,
      "Loss/HJB_replay": 0.35115426778793335,
      "Loss/BDR": 0.11174584180116653,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0010000000474974513,
      "Weights/BDR": 0.0010000000474974513
    }
  },
  {
    "name": "hold0002_R1",
    "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_001/cont0001to0002/train/0627_0211_cont0001to0002_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/ubuntu-root/hoshino/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_000/hold0002_R1/train/0630_0945_hold0002_R1_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "hold0002_R1",
      "start_checkpoint": "scheduling_experiment/fixed2randT_replayHJB_5Mto10M_restart/round_001/cont0001to0002/train/0627_0211_cont0001to0002_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.002,
        0.002
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
        "mean_abs_mc_v": 0.1217,
        "max_abs_mc_v": 0.9961,
        "mean_mc": 0.4112,
        "mean_v": 0.2896
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.0938,
        "max_abs_mc_v": 0.768,
        "mean_mc": 0.5781,
        "mean_v": 0.4902
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.2199999988079071,
      "Loss/RL": 0.007662197574973106,
      "Loss/HJB": 0.8495813608169556,
      "Loss/HJB_replay": 0.2290845811367035,
      "Loss/BDR": 0.11006398499011993,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.0020000000949949026,
      "Weights/BDR": 0.0020000000949949026
    }
  }
]
