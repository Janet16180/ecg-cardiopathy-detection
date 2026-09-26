# Experiment 018 v3 training result

The matched compact CPC continuation completed on 26 September 2026. Both
arms started from the same experiment-004 encoder and seed-18042 prediction
heads, used the same objective, optimizer, 128-record batches, historical
train-only normalization and **902 updates / 115,359 record exposures**.
The old control sampled the 56,875-record pool with replacement. The new arm
visited each record of the seed-shuffled 115,359-record cohort once. No
supervised target was exposed during this pretraining.

| Arm | Updates | Exposures | Training seconds | Mean CPC loss | Final checkpoint SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Old-pool control | 902 | 115,359 | 361.21 | 2.35513 | `5275ec6199433fbc7f779d6fe0e0dc433291f23c86c403d13cace873ee6ac91d` |
| New cohort | 902 | 115,359 | 347.26 | 2.43679 | `a8a1e79187ba8c74c7f2cf9e535efcb8e67ce943c4e60f5fabd6a9d262521c51` |

The separately frozen [v3 protocol](experiment-018-cpc-data-scaling-v3.md)
passed its real-data V100 cost gate at **3,654.68 projected seconds** against
7,200. The profile receipt SHA-256 is
`eaf9125f1bf0f22a319dc4cb835fb88ba9898dc764fd286c59788c7cfee265ab`.
The production run launched at 03:14:33 UTC, passed a new full-cache hash,
completed the old arm at 03:36:04 and the new arm at 03:41:53 UTC. The GPU
was idle after completion. Local checkpoint receipts have SHA-256
`f236e03b008ce35548e3dd0fce692abf9d1b1ee2baf93be1a4ecd14b40841f08`
and `4d7e33a261e379f23590cd0404e06953df5d12b8fd5a76499dcfa3c67a1acbc7`
for old and new arms. The [training audit](../outputs/experiment018_cpc_data_scaling_v3/training_audit.json),
SHA-256 `45f8d2e1459fda371b20481e5d07dae1c34b0ce627306ab9d8a8f907d6796748`,
verified artifact hashes, all 902 optimizer steps, finite states, RNG
checkpoints, and changes to all 24 encoder tensors in each arm.

The new-data arm's higher training loss is **not** evidence of a worse
representation: it saw different ECGs and a different source mix. This study
has no development AUROC or AP yet. No calibration or test outcome was read.
The next useful check is a separately frozen readout on identical PTB training
labels and development patients, using one fixed classifier recipe for both
encoders and the unchanged starting encoder.
