# Experiment 038: conditional 50k provenance supplement

Frozen 30 September 2026 before the 25k development outcome or any 50k execution. This supplement
extends the [original protocol](experiment-038-cpc-xlstm.md) and its
[native GRU successor](experiment-038-cpc-xlstm-v2.md) without changing the scientific comparison,
decision rule, seeds, exposures, architecture, optimizer, normalizer or readout.

Read-only review found that the successor checks the audited 25k trigger each time, but its generic
identity does not bind the 25k prediction arrays used in the 50k scaling contrast. A new, 50k-only
wrapper adds the 25k manifest, result, independent audit and development-prediction SHA-256 hashes
to the immutable 50k manifest. Every later stage must match them. It retains the successor output
namespace `outputs/experiment038_cpc_xlstm_v2/`, keeping the unchanged 25k reference and placing the
new, uniquely identified 50k execution under its unused `50k/` directory. The wrapper sources, CLI
and this committed protocol are additional pinned inputs. Original executed files remain unchanged.

The wrapper also requires the latest recorded training attempt to have completed successfully and
charged tier work to remain below 7,200 seconds before readout. This prevents a completed checkpoint
from being scored after a final training resource check failed. The running agent applies the same
successful-stage and elapsed-time check manually before 25k readout; it does not rewrite that manifest.

50k is permitted only after the unchanged, independently audited 25k result sets `run_50k=true`.
Both models then train fresh for 250,000 exposures each, exactly as frozen. Its separate 7,200-second
ceiling charges actual cache construction, preparation, real GPU profile, training, readout and audit;
the conservative projection and pace guards remain required. No development score is authorized
before a passed real GPU profile and exact checkpoint recovery for both arms. No closed evaluation
access or additional tuning is authorized. Report this provenance supplement alongside the combined
038 outputs and rank further work in the backlog.
