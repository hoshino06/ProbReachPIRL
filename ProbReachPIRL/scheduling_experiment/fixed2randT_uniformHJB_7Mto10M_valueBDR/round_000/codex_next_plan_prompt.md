You are controlling the next round of PIRL weight and collocation-distribution scheduling.

Objective:
- Treat 10000000 total updates as the first milestone, not a hard stop.
- By that milestone, outperform the TD3 baseline from scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003/hold0025_from6M/train/0627_1812_hold0025_from6M_seed_1/ckpt-7000000.
- If reward and MC reachability remain stable, keep progressing beyond the milestone.
- Keep final reward no worse than TD3 while reducing value calibration error mean|MC-V|.
- For replay_expand experiments, start close to replay-HJB and expand the HJB collocation neighborhood cautiously.

Output:
- Write ONLY valid JSON to: /home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_7Mto10M_valueBDR/round_001_plan.json
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
      "schedule_sharpness": 1e-5,
      "pinn_sample_mode": "replay_expand",
      "pinn_replay_fraction": 1.0,
      "pinn_replay_jitter": 0.0,
      "pinn_expand_jitter_initial": 0.0,
      "pinn_expand_jitter_final": 0.05,
      "pinn_expand_center": 500000,
      "pinn_expand_sharpness": 1e-5,
      "pinn_expand_time_base": "local"
    }
  ]
}

Selection rules:
- Continue from the best safe checkpoint when reward and MC are stable.
- If reward or meanMC degraded, reduce weights or slow the schedule before trying larger weights.
- Increase HJB/BDR gradually.
- Increase replay_expand jitter gradually; prefer holding or backing off jitter before increasing HJB/BDR when reward or meanMC weakens.
- Candidate-level pinn_* fields are optional; omitted fields inherit [training_env] defaults from the TOML config.
- Use at most one TD3-restart control per round, unless all scheduling checkpoints collapsed.
- Do not repeat an existing start_checkpoint + schedule_initial + schedule_final + pinn_expand_jitter_final combination unless round_note explains why.

Advisor context from TOML:

Manual notes:
- Goal: increase mean V / reduce mean|MC-V| while preserving the 7M mean MC level.
- Prefer small HJB with higher BDR; avoid increasing HJB aggressively if mean MC drops.
- Baseline 7M hold0025_from6M: avg meanMC about 0.487, avg meanV about 0.337, avg mean|MC-V| about 0.154.
- Treat 10M as a milestone, not a hard stop. If a branch is healthy, continuing beyond 10M within the 5 rounds is acceptable.
- Branching back to the best 7M/8M/9M checkpoint is acceptable if a later continuation improves mean V but hurts mean MC or calibration.

Reference paths:
- scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/all_results.json
- scheduling_experiment/fixed2randT_uniformHJB_5Mto10M.toml


Completed results JSON:
[
  {
    "name": "hjb0025_bdr0035",
    "start_checkpoint": "scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003/hold0025_from6M/train/0627_1812_hold0025_from6M_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_7Mto10M_valueBDR/round_000/hjb0025_bdr0035/train/0702_0329_hjb0025_bdr0035_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "hjb0025_bdr0035",
      "start_checkpoint": "scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003/hold0025_from6M/train/0627_1812_hold0025_from6M_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.025,
        0.025
      ],
      "schedule_final": [
        1.0,
        0.025,
        0.035
      ],
      "schedule_center": 300000,
      "schedule_sharpness": 1.5e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1708,
        "max_abs_mc_v": 0.999,
        "mean_mc": 0.4007,
        "mean_v": 0.2322
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.147,
        "max_abs_mc_v": 0.9463,
        "mean_mc": 0.5688,
        "mean_v": 0.426
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.10999999940395355,
      "Loss/RL": 0.006595359183847904,
      "Loss/HJB": 0.007935818284749985,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.023614218458533287,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.02500000037252903,
      "Weights/BDR": 0.03500000014901161
    }
  },
  {
    "name": "hjb002_bdr004",
    "start_checkpoint": "scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003/hold0025_from6M/train/0627_1812_hold0025_from6M_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_7Mto10M_valueBDR/round_000/hjb002_bdr004/train/0702_0329_hjb002_bdr004_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "hjb002_bdr004",
      "start_checkpoint": "scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003/hold0025_from6M/train/0627_1812_hold0025_from6M_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.025,
        0.025
      ],
      "schedule_final": [
        1.0,
        0.02,
        0.04
      ],
      "schedule_center": 300000,
      "schedule_sharpness": 1.5e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1733,
        "max_abs_mc_v": 0.996,
        "mean_mc": 0.3998,
        "mean_v": 0.2302
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1513,
        "max_abs_mc_v": 0.9179,
        "mean_mc": 0.5719,
        "mean_v": 0.4242
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.2199999988079071,
      "Loss/RL": 0.006687768269330263,
      "Loss/HJB": 0.025349706411361694,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.005120047368109226,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.019999999552965164,
      "Weights/BDR": 0.03999999910593033
    }
  },
  {
    "name": "cont_hjb002_bdr0035",
    "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_7Mto10M_valueBDR/round_000/hjb002_bdr004/train/0702_0329_hjb002_bdr004_seed_1/ckpt-8000000",
    "start_itr": 8000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_7Mto10M_valueBDR/round_001/cont_hjb002_bdr0035/train/0702_2014_cont_hjb002_bdr0035_seed_1/ckpt-9000000",
    "target_updates": 9000000,
    "candidate": {
      "name": "cont_hjb002_bdr0035",
      "start_checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_7Mto10M_valueBDR/round_000/hjb002_bdr004/train/0702_0329_hjb002_bdr004_seed_1/ckpt-8000000",
      "schedule_initial": [
        1.0,
        0.02,
        0.04
      ],
      "schedule_final": [
        1.0,
        0.02,
        0.035
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.177,
        "max_abs_mc_v": 0.998,
        "mean_mc": 0.4002,
        "mean_v": 0.2247
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1466,
        "max_abs_mc_v": 0.9457,
        "mean_mc": 0.5711,
        "mean_v": 0.4284
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.27000001072883606,
      "Loss/RL": 0.006542677525430918,
      "Loss/HJB": 0.007030017673969269,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.004107494372874498,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.019999999552965164,
      "Weights/BDR": 0.03500000014901161
    }
  },
  {
    "name": "hjb0025_bdr003",
    "start_checkpoint": "scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003/hold0025_from6M/train/0627_1812_hold0025_from6M_seed_1/ckpt-7000000",
    "start_itr": 7000000,
    "checkpoint": "/home/ubuntu/ProbReachPIRL/ProbReachPIRL/scheduling_experiment/fixed2randT_uniformHJB_7Mto10M_valueBDR/round_001/hjb0025_bdr003/train/0702_2014_hjb0025_bdr003_seed_1/ckpt-8000000",
    "target_updates": 8000000,
    "candidate": {
      "name": "hjb0025_bdr003",
      "start_checkpoint": "scheduling_experiment/fixed2randT_uniformHJB_5Mto10M/round_003/hold0025_from6M/train/0627_1812_hold0025_from6M_seed_1/ckpt-7000000",
      "schedule_initial": [
        1.0,
        0.025,
        0.025
      ],
      "schedule_final": [
        1.0,
        0.025,
        0.03
      ],
      "schedule_center": 500000,
      "schedule_sharpness": 1e-05
    },
    "mc_metrics": {
      "beta_r": {
        "mean_abs_mc_v": 0.1707,
        "max_abs_mc_v": 0.9976,
        "mean_mc": 0.4029,
        "mean_v": 0.2333
      },
      "ey_epsi": {
        "mean_abs_mc_v": 0.1472,
        "max_abs_mc_v": 0.9251,
        "mean_mc": 0.5723,
        "mean_v": 0.4291
      }
    },
    "tensorboard_last": {
      "RL/Average Reward": 0.18000000715255737,
      "Loss/RL": 0.006018847692757845,
      "Loss/HJB": 0.01203389372676611,
      "Loss/HJB_replay": null,
      "Loss/BDR": 0.013185283169150352,
      "Weights/RL": 1.0,
      "Weights/HJB": 0.02500000037252903,
      "Weights/BDR": 0.029999999329447746
    }
  }
]
