# Pretrained ECG generator source and release assessment

18 October 2026. Pulse2Pulse is the only inspected release ready for a small, separately frozen
local CPU loading/generation check within the current storage allowance. This is source/metadata
inspection, not an inference result. No weights were downloaded, no remote code was executed, no
packages were installed, and no project ECG or report was sent to an external service.

None of these four release interfaces establishes regional disease truth. A class or text condition
requests a whole ECG distribution; it does not establish which lead-time cells express a disease.
Changing a latent, prompt or seed can alter the whole signal. An original-minus-generated
comparison would measure an intervention difference, not a clinically verified abnormality mask.
Generated backgrounds would therefore remain unlabeled until a distinct truth policy is defined.
No memorization or membership audit was performed. The authors' synthetic-data privacy language
does not establish that released outputs cannot reproduce training records.

The latest audited paired-loss study remains a positive engineering result with a repeated-edit
tradeoff, single-fit uncertainty and no clinical validation. A new generator can broaden background
variation; merely loading it cannot reproduce or improve that localization result.

## Ranking for this machine and task

| Rank | Candidate | Verified release evidence | Regional control | Present decision |
|---|---|---|---|---|
| 1 | Pulse2Pulse / DeepFake ECG | Hub declares a 42,375,017-byte model file and SHA-256; small author architecture available | Unconditional noise-to-full-ECG, no edit mask | Freeze a tiny CPU technical check; no clinical labels |
| 2 | SSSD-ECG | Author V1.1 links a Figshare model; current API lookup did not yield a downloadable file manifest | Global 71-label conditioning in supplied sampler, no observed-region mask | Resolve model size/hash/access first; dependency and overlap blockers remain |
| 3 | DiffuSETS | Author Hub lists a 1,485,588,360-byte inference archive with SHA-256 | Whole-record text/age/sex/heart-rate conditioning | Does not fit remaining storage; standard demo also makes an external embedding call |
| 4 | cNVAE-ECG | Author source and training instructions available; no checkpoint file/release asset or weight link found | Global label vector/temperature, no regional edit mask | Not a verified pretrained option; license and checkpoint provenance unresolved |

This ranking concerns bounded readiness, not physiological quality or a GAN-versus-diffusion-versus-VAE
comparison. No measured speed, GPU peak memory, output realism or downstream performance is claimed.

## Pulse2Pulse: exact proposed asset and local API

The [author Hub](https://huggingface.co/deepsynthbody/deepfake_ecg/tree/262e0e67c0101cd4989d7c2f66f16d6d54756e70)
was inspected through its public model/tree metadata. Immutable revision:
`262e0e67c0101cd4989d7c2f66f16d6d54756e70`.

| Asset | Size | SHA-256 |
|---|---:|---|
| `pytorch_model.bin` | 42,375,017 bytes | `e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827` |
| `modeling_deepfake.py` | 11,269 bytes | `160fdee62b49e9960c09d2e669bb13c8069668930ae26019e63bd242c9f6a5b5` |
| `configurations_deepfake.py` | 641 bytes | `e079ef58129012ae07785952a8dde0e974741f25d6649bbbce548e9cebc5dd01` |
| `config.json` | 287 bytes | `8b5b15e0f3fe72baea0e20f636b7a9d13d53a4276f40ea38ae2417846a703700` |

The weight hash/size are **Hub LFS declarations**, not locally verified payloads. The three small
source/config files were actually retrieved and hashed. Direct fixed URLs are
[weights](https://huggingface.co/deepsynthbody/deepfake_ecg/resolve/262e0e67c0101cd4989d7c2f66f16d6d54756e70/pytorch_model.bin),
[architecture/wrapper](https://huggingface.co/deepsynthbody/deepfake_ecg/resolve/262e0e67c0101cd4989d7c2f66f16d6d54756e70/modeling_deepfake.py)
and [configuration](https://huggingface.co/deepsynthbody/deepfake_ecg/resolve/262e0e67c0101cd4989d7c2f66f16d6d54756e70/config.json).

The generator is a U-Net-like 1D convolutional network. Its input is uniform noise in `[-1,1]`,
shape `(batch,8,5000)`; its tanh output has that same shape. The Hub wrapper uses model size 50,
upsampling enabled, eight channels, and an unused compatibility linear layer of 100 to 500 units.
It returns a list of `(5000,8)` integer arrays, generated sequentially. The source's `labels`
argument is unused. No input ECG, clinical report, external endpoint or conditioning model is
needed by that forward path.

The [model card](https://huggingface.co/deepsynthbody/deepfake_ecg/blob/262e0e67c0101cd4989d7c2f66f16d6d54756e70/README.md)
specifies independent leads I, II, V1–V6 and 10 seconds at 500 Hz. Canonical 12-lead output requires
explicit reordering and derived III, aVR, aVL and aVF. The numerical wrapper applies a factor of
6,000, truncates to integer and transposes (source lines 293–297). That proves its export arithmetic,
not a physical calibration. A later check can retain both raw tanh output and the exact author
integer export. Treating those integer units as microvolts and dividing by 1,000 to label mV remains
a declared convention pending clearer checkpoint-specific calibration evidence.

The [paper](https://www.nature.com/articles/s41598-021-01295-2) reports 7,233 ECGs from GESUS and
Inter99 selected using MUSE 12SL's normal classification. These are different declared cohorts from
our PTB-XL backgrounds, but there is no released per-record manifest proving this particular Hub
checkpoint's cohort, epoch or absence of project-patient overlap. Machine-normal selection is not
verified healthy truth. The publication's generated amplitude summaries do not alone certify the
units of this specific binary.

### License and implementation route

The Hub metadata declares generic `bsd`, while the card's License section links MIT; it contains no
full license file. This is inconsistent release information. The current
[packaged repository](https://github.com/vlbthambawita/deepfake-ecg/tree/1a80d80e867516ccfcafdb4b80edcc85effd42a6)
is internally inconsistent: its `pyproject.toml` declares MIT while `deepfakeecg.py` has a GPLv3+
header. It also changed the default numeric scale to 6.0, while export/plot comments retain
microvolt wording. Do not silently mix that package with the fixed Hub wrapper.

A clearer architecture source is the author's original
[Pulse2Pulse development repository](https://github.com/vlbthambawita/Pulse2Pulse/tree/f28e4b792e79c503b8b7e0912be9d304a3a7177b),
which has an explicit [MIT license](https://github.com/vlbthambawita/Pulse2Pulse/blob/f28e4b792e79c503b8b7e0912be9d304a3a7177b/LICENSE).
Its `models/pulse2pulse.py` is 10,390 bytes with SHA-256
`93273ea470c82cd90d7f705c1000e464bfa3e16f7d7733bb817c5bb6cbdbe78b`;
the license SHA-256 is `8f56e0a848cdb298c9b997c7d97188ebddb3d123c857684c660b6661abece712`.
Static AST comparison found both transpose-layer classes identical to the Hub definitions. After
renaming the historical `WaveGANGenerator` class, the only generator differences are the Hub's
unused latent-dimension attribute and compatibility linear layer; the forward computation matches.
A locally attributed adapter can use the MIT architecture plus this explicit compatibility layer.
The weight-card license remains a declaration with incomplete terms. No weights, copied package
wrapper or generated clinical examples should be redistributed by this feasibility check.

Proposed minimal callable boundary after a committed protocol is:
`load_generator(path) -> nn.Module`, requiring actual file SHA/size and strict all-tensor
`torch.load(..., map_location='cpu', weights_only=True)`; and
`generate(generator, seed) -> raw8, integer_export8, derived12` with a local CPU RNG. Reject an
unexpected object, key, tensor shape, dtype or nonfinite state rather than falling back to unsafe
pickle. No `AutoModel`, `trust_remote_code`, unrestricted `torch.load`, network inference or automatic
package installation is required. The current environment already has PyTorch 2.6.0, NumPy 2.4.6 and
SciPy 1.17.1. Transformers is absent and unnecessary for the proposed plain-Torch route.

## SSSD-ECG

The [author repository](https://github.com/AI4HealthUOL/SSSD-ECG/tree/324f908eec20890768669f64bce0591db9b8a5f4)
has an explicit MIT software license. Its
[V1.1 README](https://github.com/AI4HealthUOL/SSSD-ECG/blob/V1.1/README.md) links the model at
[Figshare shared release](https://figshare.com/s/81834b24a4711c2a5c55), DOI
`10.6084/m9.figshare.21922875`. During inspection the shared URL returned HTTP 202 with an empty
body and the public article API returned 404. GitHub releases V1.0/V1.1 had no binary assets. This
is an unresolved access result, not proof that no weights exist; no size/hash/license for a model
payload was verified and no download was attempted.

The supplied [configuration](https://github.com/AI4HealthUOL/SSSD-ECG/blob/324f908eec20890768669f64bce0591db9b8a5f4/src/sssd/config/config_SSSD_ECG.json)
uses eight independent channels, 1,000 samples/10 seconds at 100 Hz, 71 label inputs, 36 residual
layers/256 channels and 200 diffusion steps. Its sampler derives twelve leads after generation.
The preprocessing notebook splits PTB-XL by its original folds, with folds below 9 in training.
That makes source-population overlap with our PTB-XL development patients plausible; exact released
checkpoint membership is not established. A class request is not a localized annotation.

The current inference script hardcodes CUDA and reads a test-label array. We would not run it
unchanged. Its sampling signature takes labels, not a measured waveform and regional mask. S4 code
imports Lightning, einops, opt_einsum and optional Cauchy/KeOps acceleration; some are absent from the
pinned environment. No 3 GB GPU or CPU profile has been measured. Physical output units require
checkpoint/preprocessing verification rather than assuming that a config's rate establishes mV.

## DiffuSETS

The [author experiment repository](https://github.com/Raiiyf/DiffuSETS_Exp/tree/f9106528a9e5348d94a118a71eb79d459ee4197a)
and [official demo](https://github.com/PKUDigitalHealth/DiffuSETS/tree/c21c74bb6f4225969634edbd024ce5ed79eddf83)
point to [Laiyf/DiffuSETS](https://huggingface.co/Laiyf/DiffuSETS). Hub revision
`80b04013aef594c7c3012a8b932e5e09524e581a` declares MIT and lists the misspelled actual filename
`prerequsites_for_inference.zip`: 1,485,588,360 bytes, SHA-256
`3402ccad9459bd8e03d55f031dbcf02e65c9e5d5deee706c33ab39c7dbabf1e9`.
This compressed archive alone exceeds the 367,409,581-byte remaining allowance. The separate
training latent ZIP is 5,913,552,037 bytes and is not needed or appropriate here. The card says the
inference archive also contains MIMIC/PTB-XL latents, classification data and embeddings, not only
weights. Those clinical-derived artifacts must not be casually downloaded as model dependencies.

The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12546759/) trains on MIMIC-IV-ECG reports and
waveforms. Such machine report conditions are not trusted clinical labels here. The decoder emits
12 channels/1,024 samples for ten seconds (102.4 Hz), through a four-channel/128-position latent;
the demonstration performs 1,000 diffusion steps. Measured mV calibration and cross-lead constraints
have not been independently established for the released checkpoint.

The standard demo's [embedding helper](https://github.com/PKUDigitalHealth/DiffuSETS/blob/c21c74bb6f4225969634edbd024ce5ed79eddf83/utils/text_to_emb.py)
POSTs the input text to an external embedding service. The experiment README also describes an API
route. Neither may receive our reports. Its alternate public precomputed-embedding route can avoid
that call, but does not solve storage, private-latent bundling, checkpoint provenance or conditioning
truth. Dependencies include diffusers and einops (absent here), plus a much larger historical
training environment. No 3 GB GPU feasibility is established. This full-record text generator has
no exact regional-control contract.

## cNVAE-ECG

The [author repository](https://github.com/univanxx/cNVAE_ECG/tree/b50664c2f43937605159926550e8bcfd6731c15c)
contains source and training/generation instructions. Its full recursive tree has no checkpoint,
weight-download link or license file; the release API returned no releases and repository metadata
reported no recognized license. This inspection found no pretrained asset with a verifiable size,
hash or inference license. Do not confuse publicly available implementation with available weights.

The supplied training recipe uses PTB-XL, Ningbo and Georgia, 5,000 samples, eight independent
channels and global diagnosis labels. The loader selects I, III and V1–V6, then reconstructs II and
the augmented leads; this is a different independent-lead convention from Pulse2Pulse. Signals are
min-max normalized with cohort-derived arrays, which are also required to recover output scale.
Without the checkpoint and matching normalization arrays, its physical output scale is unresolved.

The generation script uses unrestricted `torch.load(..., weights_only=False)`, serialized training
arguments, CUDA/distributed setup and optional batch-normalization readjustment. It cannot be used
unchanged under a strict all-tensor local-loader contract. Its requirements include fastai 1.x,
inherited NVAE training infrastructure, torchvision, OpenCV and other packages; a compatible
minimal inference environment and 3 GB memory fit are unverified. This is a research implementation
candidate, not a ready local pretrained model, and its source populations overlap ours.

## Bounded next check, requiring a frozen protocol

A technical Pulse2Pulse probe can use four fixed seeds, batch size one and one exact replay, with no
input ECG or text. Freeze source revisions, expected binary SHA/size, strict loader rules and the
unit limitation before download. Check deterministic finite output, native geometry, diversity,
nonzero output, exact wrapper truncation and algebraic lead reconstruction. These are technical
gates, not realism, healthy status, disease conditioning or localization gates. Record raw and
export units explicitly; do not manufacture a physical-unit PASS.

A 96 MiB new-study cap is plausible: the 42,375,017-byte checkpoint, four float32 12-lead outputs
(960,000 bytes), four noise arrays (640,000 bytes), small public sources/receipts and the remainder as
reserve fit below 100,663,296 bytes. Even a 128 MiB conservative reservation fits the current whole
stream under 4 GiB. CPU and monotonic wall ceilings can both be 180 seconds with one numerical and
interop thread, a live deadline, no GPU and no retries. These are prospective bounds; generation
runtime and RAM remain unmeasured. Existing stream receipts are not to be deleted or overwritten.

No inference or weight download is authorized by this assessment itself. The coordinator must first
commit the small prospective protocol, then integrate/review the new adapter and artificial tests.
All actual outputs and a source-bound receipt would remain local. Any future performance comparison
still needs cohort/overlap admission, a new truth policy and a separate prospective study.
