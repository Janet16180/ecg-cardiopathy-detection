# Challenge SNOMED codes to the binary endpoint, version 1

29 September 2026. `ecg_experiment/challenge_labels.py` maps the SNOMED CT codes of the PhysioNet/CinC
Challenge sources (Chapman/Shaoxing, Ningbo, Georgia, CPSC 2018 and CPSC-Extra) to the binary label used for
PTB-XL and SPH ([Experiment 022](experiment-022-sph-external-readout.md)). The code is tested in
`tests/test_challenge_labels.py`. No model was trained.

## Official tables

Code names, the scored/unscored split and the scoring equivalences come from the official tables of the
[Challenge 2021 evaluation repository](https://github.com/physionetchallenges/evaluation-2021), fetched once on
29 September 2026 at commit `e2a75fc01f729cb74cc4e853e054ce81e28381fc`. The copies are in
`data/raw/challenge-2021/evaluation-2021/` (outside Git), and the module refuses to run if either differs:

| File | SHA-256 |
| --- | --- |
| `dx_mapping_scored.csv` | `fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0` |
| `dx_mapping_unscored.csv` | `ce53c9e35406e922f1634a38bb13c31e8cf7603291641a16c201d8e679f4fe94` |

```bash
mkdir -p data/raw/challenge-2021/evaluation-2021 && cd data/raw/challenge-2021/evaluation-2021
for f in dx_mapping_scored.csv dx_mapping_unscored.csv; do
  curl -sSfLO https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/$f
done
```

The four equivalences are read from the `Notes` column of `dx_mapping_scored.csv` ("We score X and Y as the
same diagnosis"), not from a hand-written list; the first code of each note is the canonical one:
713427006 ≡ 59118001 (RBBB), 733534002 ≡ 164909002 (LBBB), 284470004 ≡ 63593006 (premature atrial or
supraventricular beats), 427172004 ≡ 17338001 (premature ventricular beats). Both codes of every pair are in the
same endpoint group, which the tests check.

## Rule

| Label | Definition |
| --- | --- |
| Positive (1) | any MI, STTC, CD or HYP code, whatever else is listed |
| Primary negative (0) | sinus rhythm (426783006) is the only code |
| Secondary negative (0) | only sinus rhythm and the benign sinus variants: bradycardia (426177001), tachycardia (427084000), arrhythmia (427393009) |
| Undefined (NaN) | anything else, including any code outside the groups below and any code in neither official table |

The Challenge has no "normal ECG" statement: 426783006 is a rhythm. Sinus rhythm alone is the closest
equivalent of a normal ECG, because the readers list abnormal findings as further codes. The secondary label
mirrors SPH's, where an otherwise normal ECG with a sinus variant is coded by the rhythm alone. Codes outside
the four superclasses, such as atrial fibrillation or flutter, premature beats, axis deviation, low voltage,
paced rhythm or early repolarization, are rhythm and form statements that PTB-XL and SPH ignore; a record whose
only findings are such codes stays undefined, never negative.

Three choices follow PTB-XL's split between diagnostic and form statements:

- Left ventricular high voltage (55827005) is a voltage criterion, like PTB-XL's `HVOLT` form statement, not
  the diagnosis of hypertrophy (164873001); it is `other`. It is common in Ningbo (4,106 records) and Chapman
  (1,295), so this choice moves many records to undefined.
- Abnormal Q waves (164917005) and poor R-wave progression (365413008) are form statements, not infarction
  diagnoses; they are `other`.
- The ischemia statements (myocardial ischemia and the anterior, inferior and lateral ischemia codes) are STTC,
  as PTB-XL's `ISC_` statements are. The generic ST and T statements (ST depression and elevation, T-wave
  abnormality and inversion, "s t changes", nonspecific ST-T abnormality, prolonged QT) are STTC, as SPH's
  codes 145-148 are.

Alternatives for the user, not applied: (1) count left ventricular high voltage as HYP, which would make about
1,900 more Ningbo records positive; (2) count the loop-rotation codes (61721007, 251198002, 251199005), which
describe a normal variant of the frontal axis, as allowed with sinus rhythm in the secondary label. Both would
need a new version of the mapping.

## Labels per source

Counted on every record with a signal file (Chapman 10,247 before its strict 10 s view; Ningbo 34,903 before
cleaning):

| Source | Records | Primary negative | Positive | Primary undefined | Secondary negative | Secondary undefined | MI | STTC | CD | HYP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chapman | 10,247 | 1,366 | 3,730 | 5,151 | 4,212 | 2,305 | 40 | 2,951 | 1,226 | 21 |
| Ningbo | 34,903 | 4,542 | 12,601 | 17,760 | 15,639 | 6,663 | 165 | 10,407 | 3,893 | 749 |
| Georgia | 10,344 | 1,752 | 6,959 | 1,633 | 2,607 | 778 | 7 | 5,011 | 2,238 | 2,216 |
| CPSC 2018 | 6,877 | 918 | 3,828 | 2,131 | 918 | 2,131 | 0 | 1,087 | 2,797 | 0 |
| CPSC-Extra | 3,453 | 0 | 3,370 | 83 | 2 | 81 | 1,515 | 2,349 | 379 | 225 |

- CPSC 2018 annotates only nine classes and no sinus variants, so its primary and secondary labels are equal.
- CPSC-Extra holds records that were not normal, and its 4 sinus-rhythm records all carry another code.
- Georgia has by far the most HYP (2,216 records), mostly left ventricular hypertrophy (1,233) and left atrial
  enlargement (870).
- Ningbo codes no atrial fibrillation at all and atrial flutter on 21.8% of records
  ([notebook 09](../notebooks/09-jr-ningbo.ipynb)); neither code affects the binary label.

## Mapping table

Every code used by at least one source, with its group and records per source (a record counts once per
code). "Canonical" names the code it is scored as, where an equivalence applies. The 15 official codes that
none of these sources uses are mapped by the same rule (all to `other`).

| Code | Name | Group | Scored | Canonical | Chapman | Ningbo | Georgia | CPSC 2018 | CPSC-Extra |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 426783006 | sinus rhythm | sinus_rhythm | yes |  | 1,826 | 6,299 | 1,752 | 918 | 4 |
| 426177001 | sinus bradycardia | sinus_variant | yes |  | 3,889 | 12,670 | 1,677 | 0 | 45 |
| 427084000 | sinus tachycardia | sinus_variant | yes |  | 1,568 | 5,687 | 1,261 | 0 | 303 |
| 427393009 | sinus arrhythmia | sinus_variant | yes |  | 0 | 2,550 | 455 | 0 | 11 |
| 164867002 | old myocardial infarction | MI | no |  | 0 | 0 | 0 | 0 | 1,168 |
| 164865005 | myocardial infarction | MI | no |  | 40 | 83 | 7 | 0 | 376 |
| 54329005 | anterior myocardial infarction | MI | no |  | 0 | 57 | 0 | 0 | 62 |
| 57054005 | acute myocardial infarction | MI | no |  | 0 | 49 | 0 | 0 | 0 |
| 164934002 | t wave abnormal | STTC | yes |  | 1,876 | 5,167 | 2,306 | 0 | 22 |
| 428750005 | nonspecific st t abnormality | STTC | no |  | 1,158 | 0 | 1,883 | 0 | 1,293 |
| 55930002 | s t changes | STTC | no |  | 0 | 4,232 | 6 | 0 | 1 |
| 59931005 | t wave inversion | STTC | yes |  | 157 | 2,720 | 812 | 0 | 5 |
| 429622005 | st depression | STTC | no |  | 402 | 1,266 | 38 | 869 | 57 |
| 164930006 | st interval abnormal | STTC | no |  | 2 | 799 | 992 | 0 | 481 |
| 111975006 | prolonged qt interval | STTC | yes |  | 57 | 337 | 1,391 | 0 | 4 |
| 425623009 | lateral ischaemia | STTC | no |  | 0 | 0 | 905 | 0 | 0 |
| 164931005 | st elevation | STTC | no |  | 176 | 0 | 134 | 220 | 66 |
| 425419005 | inferior ischaemia | STTC | no |  | 0 | 0 | 451 | 0 | 0 |
| 164861001 | myocardial ischemia | STTC | no |  | 0 | 0 | 0 | 0 | 384 |
| 426434006 | anterior ischemia | STTC | no |  | 0 | 0 | 281 | 0 | 0 |
| 413844008 | chronic myocardial ischemia | STTC | no |  | 0 | 0 | 0 | 0 | 161 |
| 413444003 | acute myocardial ischemia | STTC | no |  | 0 | 0 | 1 | 0 | 1 |
| 704997005 | inferior ST segment depression | STTC | no |  | 0 | 0 | 0 | 0 | 1 |
| 370365005 | left ventricular strain | STTC | no |  | 0 | 0 | 0 | 0 | 1 |
| 59118001 | right bundle branch block | CD | yes | 713427006 | 454 | 195 | 542 | 1,857 | 1 |
| 270492004 | 1st degree av block | CD | yes |  | 247 | 893 | 769 | 722 | 106 |
| 713427006 | complete right bundle branch block | CD | yes |  | 0 | 1,096 | 28 | 0 | 113 |
| 698252002 | nonspecific intraventricular conduction disorder | CD | yes |  | 235 | 536 | 203 | 0 | 4 |
| 164909002 | left bundle branch block | CD | yes | 733534002 | 205 | 35 | 231 | 236 | 38 |
| 713426002 | incomplete right bundle branch block | CD | yes |  | 0 | 246 | 407 | 0 | 86 |
| 445118002 | left anterior fascicular block | CD | yes |  | 0 | 380 | 180 | 0 | 0 |
| 6374002 | bundle branch block | CD | yes |  | 0 | 385 | 116 | 0 | 0 |
| 233917008 | av block | CD | no |  | 166 | 78 | 74 | 0 | 5 |
| 733534002 | complete left bundle branch block | CD | yes |  | 0 | 212 | 0 | 0 | 0 |
| 251120003 | incomplete left bundle branch block | CD | no |  | 0 | 6 | 86 | 0 | 42 |
| 27885002 | complete heart block | CD | no |  | 1 | 75 | 8 | 0 | 27 |
| 195042002 | 2nd degree av block | CD | no |  | 8 | 58 | 23 | 0 | 21 |
| 74390002 | wolff parkinson white pattern | CD | no |  | 4 | 68 | 2 | 0 | 0 |
| 164947007 | prolonged pr interval | CD | yes |  | 12 | 40 | 0 | 0 | 0 |
| 54016002 | mobitz type i wenckebach atrioventricular block | CD | no |  | 6 | 25 | 0 | 0 | 0 |
| 445211001 | left posterior fascicular block | CD | no |  | 0 | 5 | 25 | 0 | 0 |
| 195060002 | ventricular pre excitation | CD | no |  | 12 | 0 | 2 | 0 | 6 |
| 426183003 | mobitz type II atrioventricular block | CD | no |  | 0 | 7 | 0 | 0 | 0 |
| 82226007 | diffuse intraventricular block | CD | no |  | 0 | 0 | 0 | 0 | 1 |
| 164873001 | left ventricular hypertrophy | HYP | no |  | 15 | 632 | 1,233 | 0 | 158 |
| 67741000119109 | left atrial enlargement | HYP | no |  | 0 | 1 | 870 | 0 | 1 |
| 89792004 | right ventricular hypertrophy | HYP | no |  | 4 | 106 | 86 | 0 | 20 |
| 266249003 | ventricular hypertrophy | HYP | no |  | 0 | 0 | 71 | 0 | 5 |
| 253352002 | left atrial abnormality | HYP | no |  | 0 | 0 | 72 | 0 | 0 |
| 195126007 | atrial hypertrophy | HYP | no |  | 0 | 0 | 60 | 0 | 2 |
| 446358003 | right atrial hypertrophy | HYP | no |  | 3 | 33 | 0 | 0 | 18 |
| 446813000 | left atrial hypertrophy | HYP | no |  | 0 | 8 | 0 | 0 | 40 |
| 253339007 | right atrial abnormality | HYP | no |  | 0 | 0 | 14 | 0 | 0 |
| 164890007 | atrial flutter | other | yes |  | 445 | 7,614 | 186 | 0 | 54 |
| 55827005 | left ventricular high voltage | other | no |  | 1,295 | 4,106 | 0 | 0 | 0 |
| 164889003 | atrial fibrillation | other | yes |  | 1,780 | 0 | 570 | 1,221 | 153 |
| 284470004 | premature atrial contraction | other | yes |  | 258 | 1,054 | 1,236 | 616 | 73 |
| 39732003 | left axis deviation | other | yes |  | 382 | 1,162 | 940 | 0 | 0 |
| 164917005 | qwave abnormal | other | yes |  | 235 | 828 | 464 | 0 | 1 |
| 251146004 | low qrs voltages | other | yes |  | 249 | 794 | 374 | 0 | 0 |
| 427172004 | premature ventricular contractions | other | yes |  | 0 | 1,091 | 0 | 0 | 188 |
| 10370003 | pacing rhythm | other | yes |  | 0 | 1,181 | 0 | 0 | 3 |
| 47665007 | right axis deviation | other | yes |  | 215 | 638 | 83 | 0 | 1 |
| 426761007 | supraventricular tachycardia | other | no |  | 587 | 137 | 32 | 0 | 3 |
| 164884008 | ventricular ectopics | other | no |  | 0 | 0 | 42 | 700 | 0 |
| 17338001 | ventricular premature beats | other | yes | 427172004 | 294 | 0 | 387 | 0 | 8 |
| 61721007 | clockwise or counterclockwise vectorcardiographic loop | other | no |  | 0 | 653 | 0 | 0 | 0 |
| 365413008 | poor R wave Progression | other | yes |  | 0 | 638 | 0 | 0 | 0 |
| 428417006 | early repolarization | other | no |  | 22 | 344 | 140 | 0 | 0 |
| 713422000 | atrial tachycardia | other | no |  | 121 | 176 | 28 | 0 | 15 |
| 426627000 | bradycardia | other | yes |  | 0 | 7 | 6 | 0 | 271 |
| 106068003 | atrial rhythm | other | no |  | 0 | 215 | 0 | 0 | 0 |
| 251223006 | tall p wave | other | no |  | 0 | 215 | 0 | 0 | 0 |
| 251199005 | countercolockwise rotation | other | no |  | 162 | 0 | 0 | 0 | 0 |
| 29320008 | atrioventricular junctional rhythm | other | no |  | 0 | 139 | 0 | 0 | 6 |
| 164912004 | p wave change | other | no |  | 95 | 47 | 0 | 0 | 0 |
| 164937009 | u wave abnormal | other | no |  | 22 | 114 | 0 | 0 | 1 |
| 13640000 | fusion beats | other | no |  | 2 | 114 | 0 | 0 | 0 |
| 425856008 | paroxysmal ventricular tachycardia | other | no |  | 0 | 109 | 0 | 0 | 0 |
| 251205003 | prolonged P wave | other | no |  | 0 | 106 | 0 | 0 | 0 |
| 81898007 | ventricular escape rhythm | other | no |  | 0 | 96 | 1 | 0 | 1 |
| 426995002 | junctional escape | other | no |  | 15 | 60 | 5 | 0 | 4 |
| 251198002 | clockwise rotation | other | no |  | 76 | 0 | 0 | 0 | 0 |
| 164896001 | ventricular fibrillation | other | no |  | 0 | 59 | 3 | 0 | 10 |
| 251170000 | blocked premature atrial contraction | other | no |  | 0 | 62 | 0 | 0 | 2 |
| 63593006 | supraventricular premature beats | other | yes | 284470004 | 0 | 9 | 1 | 0 | 53 |
| 50799005 | atrioventricular dissociation | other | no |  | 0 | 59 | 0 | 0 | 0 |
| 75532003 | ventricular escape beat | other | no |  | 7 | 49 | 0 | 0 | 3 |
| 251268003 | atrial pacing pattern | other | no |  | 0 | 0 | 52 | 0 | 0 |
| 251266004 | ventricular pacing pattern | other | no |  | 0 | 0 | 46 | 0 | 0 |
| 195080001 | atrial fibrillation and flutter | other | no |  | 0 | 0 | 2 | 0 | 39 |
| 67751000119106 | right atrial  high voltage | other | no |  | 8 | 28 | 0 | 0 | 0 |
| 5609005 | sinus arrest | other | no |  | 0 | 33 | 0 | 0 | 0 |
| 426664006 | accelerated junctional rhythm | other | no |  | 0 | 12 | 19 | 0 | 0 |
| 426648003 | junctional tachycardia | other | no |  | 0 | 24 | 4 | 0 | 2 |
| 49578007 | shortened pr interval | other | no |  | 0 | 23 | 2 | 0 | 3 |
| 233897008 | atrioventricular reentrant tachycardia | other | no |  | 8 | 18 | 0 | 0 | 0 |
| 251187003 | atrial escape beat | other | no |  | 0 | 17 | 0 | 0 | 0 |
| 233892002 | accelerated atrial escape rhythm | other | no |  | 0 | 16 | 0 | 0 | 0 |
| 251166008 | atrioventricular  node reentrant tachycardia | other | no |  | 16 | 0 | 0 | 0 | 0 |
| 61277005 | accelerated idioventricular rhythm | other | no |  | 0 | 14 | 0 | 0 | 0 |
| 65778007 | sinoatrial block | other | no |  | 0 | 5 | 0 | 0 | 9 |
| 251164006 | junctional premature complex | other | no |  | 1 | 10 | 0 | 0 | 2 |
| 251180001 | ventricular trigeminy | other | no |  | 8 | 0 | 1 | 0 | 4 |
| 251139008 | suspect arm ecg leads reversed | other | no |  | 0 | 0 | 12 | 0 | 0 |
| 164921003 | r wave abnormal | other | no |  | 0 | 0 | 10 | 0 | 1 |
| 11157007 | ventricular bigeminy | other | no |  | 3 | 0 | 2 | 0 | 5 |
| 195101003 | wandering atrial pacemaker | other | no |  | 2 | 0 | 7 | 0 | 0 |
| 111288001 | ventricular flutter | other | no |  | 0 | 7 | 0 | 0 | 1 |
| 17366009 | sinus atrium to atrial wandering rhythm | other | no |  | 7 | 0 | 0 | 0 | 0 |
| 418818005 | brugada | other | no |  | 0 | 5 | 0 | 0 | 0 |
| 251173003 | atrial bigeminy | other | no |  | 3 | 0 | 0 | 0 | 0 |
| 164942001 | fqrs wave | other | no |  | 3 | 0 | 0 | 0 | 0 |
| 77867006 | decreased qt interval | other | no |  | 0 | 2 | 0 | 0 | 1 |
| 74615001 | brady tachy syndrome | other | no |  | 0 | 0 | 0 | 0 | 1 |
| 426749004 | chronic atrial fibrillation | other | no |  | 0 | 0 | 0 | 0 | 1 |
| 251259000 | high t-voltage | other | no |  | 0 | 0 | 0 | 0 | 1 |
| 164895002 | ventricular tachycardia | other | no |  | 0 | 0 | 0 | 0 | 1 |
