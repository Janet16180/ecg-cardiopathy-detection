# Experiment 024: review sheet for a cardiologist

About 30 to 60 minutes. Thank you for looking at these.

All ECGs are public training records from PTB-XL 1.0.3 (10 s, 12 leads, 500 Hz). They were chosen by a
computer from how an ECG model represents them, not by a person. The labels below are the PTB-XL
annotations. The report text is the original PTB-XL report (German, English or Swedish). "Human validated"
is PTB-XL's `validated_by_human` field. Codes are SCP-ECG statements with PTB-XL's likelihood (0 to 100);
a legend is at the end. Upper-case umlauts in the source reports are shown in lower case.

Please answer in the space after each question, or write on a printout. There are no wrong answers; "not
sure" is useful.

## Part 1: reference ECGs (5 questions)

For each group, the model picked the training ECG closest to the average of all ECGs in that group. We
want to know whether that ECG is a reasonable example. The plots show the raw signal with each lead
centered on its median; no filter was applied. Grid: 0.2 s by 0.5 mV.

### 1. Normal ECG (NORM only)

![NORM reference, ECG 3212](figures/experiment-024/reference_norm_3212.png)

- ECG ID 3212, device CS-12, human validated: yes
- SCP codes: NORM 100, SR 0
- Report: "sinusrhythmus lagetyp normal normales ekg"

Is this a typical example of a normal ECG? yes / no / comment:

### 2. Myocardial infarction (MI, no other superclass)

![MI reference, ECG 9600](figures/experiment-024/reference_mi_9600.png)

- ECG ID 9600, device CS100 3, human validated: yes
- SCP codes: IMI 100, ASMI 100, ABQRS 0, SR 0
- Report: "supraventrikuläre extrasystole(n) sinusrhythmus ueberdrehter linkstyp linksanteriorer
  hemiblock qrs(t) abnorm septaler infarkt nicht auszuschliessen inferiorer infarkt möglich t abnorm in
  hochlateralen ableitungen 4.46 unbestätigter bericht"
- Note: the first 0.5 s contains a large transient.

Is this a typical example of myocardial infarction? yes / no / comment:

### 3. ST/T change (STTC, no other superclass)

![STTC reference, ECG 18255](figures/experiment-024/reference_sttc_18255.png)

- ECG ID 18255, device AT-6 6, human validated: yes
- SCP codes: ISCAS 100, ISCIL 100, LOWT 0, STD_ 0, SR 0
- Report: "sinus rhythm. st segments are depressed in ii, iii, avf, v3-6. t waves are low or flat
  throughout. myocardial ischaemia is likely. the age of the changes is uncertain. suggest exclude
  hypokalaemia."

Is this a typical example of ST/T change? yes / no / comment:

### 4. Conduction disturbance (CD, no other superclass)

![CD reference, ECG 10133](figures/experiment-024/reference_cd_10133.png)

- ECG ID 10133, device AT-6 6, human validated: yes
- SCP codes: CLBBB 100, SR 0
- Report: "sinus rhythm. left bundle branch block, this is most commonly due to ischaemic heart disease."

Is this a typical example of conduction disturbance? yes / no / comment:

### 5. Hypertrophy (HYP, no other superclass)

![HYP reference, ECG 542](figures/experiment-024/reference_hyp_542.png)

- ECG ID 542, device AT-6 C, human validated: yes
- SCP codes: LVH 100, SR 0
- Report: "sinusrytm måttliga amplitudkrit. för vänster kammarkhypertrofi ospecifik st-t förändring
  (höjning)"

Is this a typical example of hypertrophy? yes / no / comment:

## Part 2: possible label errors (20 questions)

For each ECG below, at least 80% of its 25 most similar training ECGs carry the opposite binary label, in
two different ECG models (CPC and ECG-JEPA). The binary label is "NORM only" (normal) versus "abnormal"
(any MI, STTC, CD or HYP statement, whatever its likelihood). Such ECGs may be mislabeled, borderline, or
simply unusual. These are the top 20 of 192 candidates, ordered by the average of the two fractions.

The tracings are not printed here. Each one is on PhysioNet as PTB-XL
`records500/<first two digits of a five-digit ID>000/<five-digit ID>_hr`; for example ECG 2410 is
`records500/02000/02410_hr`. We can print any of them on request.

| # | ECG ID | Current label | SCP codes (likelihood) | Human validated | Neighbors disagreeing, CPC | Neighbors disagreeing, ECG-JEPA | Is the current label correct? |
| ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | 2410 | abnormal (CD) | IRBBB 80, SR 0 | yes | 1.00 | 1.00 | yes / no / comment: |
| 2 | 9669 | NORM only | NORM 100, SR 0 | no | 1.00 | 1.00 | yes / no / comment: |
| 3 | 12964 | NORM only | NORM 80, SR 0 | yes | 1.00 | 1.00 | yes / no / comment: |
| 4 | 2415 | abnormal (HYP) | SEHYP 100, ABQRS 0, SR 0 | yes | 0.96 | 1.00 | yes / no / comment: |
| 5 | 7664 | abnormal (CD) | IRBBB 100, SR 0 | yes | 1.00 | 0.96 | yes / no / comment: |
| 6 | 14915 | abnormal (CD) | IRBBB 100, SR 0 | yes | 0.96 | 1.00 | yes / no / comment: |
| 7 | 17394 | abnormal (CD+MI) | LMI 15, ASMI 50, IRBBB 100, ABQRS 0, SR 0 | yes | 1.00 | 0.96 | yes / no / comment: |
| 8 | 17624 | NORM only | NORM 100, SR 0 | yes | 1.00 | 0.96 | yes / no / comment: |
| 9 | 17820 | NORM only | NORM 100, SR 0 | yes | 1.00 | 0.96 | yes / no / comment: |
| 10 | 131 | abnormal (MI) | ASMI 50, ABQRS 0, SR 0 | yes | 0.96 | 0.96 | yes / no / comment: |
| 11 | 1636 | NORM only | NORM 100, SR 0 | yes | 0.96 | 0.96 | yes / no / comment: |
| 12 | 4796 | abnormal (MI) | ASMI 50, ABQRS 0, SR 0 | yes | 0.92 | 1.00 | yes / no / comment: |
| 13 | 10521 | abnormal (MI) | ASMI 100, SR 0 | yes | 0.92 | 1.00 | yes / no / comment: |
| 14 | 10916 | abnormal (STTC) | NDT 100, SR 0 | no | 0.96 | 0.96 | yes / no / comment: |
| 15 | 12377 | abnormal (HYP) | LVH 15, VCLVH 0, SR 0 | no | 0.96 | 0.96 | yes / no / comment: |
| 16 | 17884 | abnormal (MI) | AMI 100, ABQRS 0, SR 0 | yes | 0.96 | 0.96 | yes / no / comment: |
| 17 | 21366 | NORM only | NORM 80, SR 0 | no | 0.92 | 1.00 | yes / no / comment: |
| 18 | 2366 | abnormal (CD) | IRBBB 100, ABQRS 0, SR 0 | yes | 0.92 | 0.96 | yes / no / comment: |
| 19 | 2862 | abnormal (STTC) | NDT 100, SARRH 0 | yes | 1.00 | 0.88 | yes / no / comment: |
| 20 | 4189 | abnormal (MI) | IMI 15, ABQRS 0, SR 0 | yes | 0.88 | 1.00 | yes / no / comment: |

Original reports, with the recording device:

1. ECG 2410 (CS-12 E): sinusrhythmus lagetyp normal unvollständiger rechtsschenkelblock sonst normales ekg
2. ECG 9669 (CS100 3): sinusrhythmus lagetyp normal lagetyp in anbetracht des alters normal normales ekg
   4.46 unbestätigter bericht
3. ECG 12964 (AT-6 C 5.5): sinus rhythm. premature ventricular contractions, bigeminy. otherwise no
   definite pathology. the computer has selected vpb's as typical ventricular complexes because more of
   them have been recorded. please ignore this. Edit: BIGU
4. ECG 2415 (CS-12 E): sinusrhythmus qrs(t) abnormal inferiorer myokardschaden nicht auszuschliessen
5. ECG 7664 (CS-12 E): sinusrhythmus lagetyp normal unvollständiger rechtsschenkelblock sonst normales ekg
6. ECG 14915 (CS-12 E): sinusrhythmus lagetyp normal unvollständiger rechtsschenkelblock sonst normales
   ekg
7. ECG 17394 (CS-12): sinusrhythmus lagetyp normal unvollständiger rechtsschenkelblock qrs(t) abnormal
   hochlateraler infarkt möglich
8. ECG 17624 (AT-60 3): sinusrhythmus a-v block i verdacht auf p-sinistrocardiale linkstyp
   unspezifisches abnormes t qt-verlängerung 4.46 unbestätigter bericht
9. ECG 17820 (AT-6 C 5.5): sinus rhythm. normal ecg.
10. ECG 131 (CS-12 E): sinusrhythmus qrs(t) abnormal septaler infarkt nicht auszuschliessen
11. ECG 1636 (AT-6 C 5.8): sinus rhythm. t waves are rather low. suggest exclude hypokalaemia. no paced
    beats recorded.
12. ECG 4796 (CS-12 E): sinusrhythmus periphere niederspannung qrs(t) abnormal anteroseptaler infarkt
    wahrscheinlich alt
13. ECG 10521 (AT-6 C 5.3): sinus rhythm. qs complexes in v2, this is probably normal. no definite
    pathology. Edit: ASMI 100, probably normal (f,30), QSV(1)-2, (ASMI 15)
14. ECG 10916 (CS100 3): sinusrhythmus lagetyp normal periphere niederspannung 4.46 unbestätigter bericht
15. ECG 12377 (CS100 3): sinusrhythmus lagetyp normal mässige amplitudenkriterien für linkshypertrophie
    4.46 unbestätigter bericht
16. ECG 17884 (CS-12 E): sinusrhythmus lagetyp normal qrs(t) abnorm anteroseptaler myokardschaden nicht
    auszuschliessen
17. ECG 21366 (CS100 3): sinusrhythmus linkstyp sonst normales ekg 4.46 unbestätigter bericht
18. ECG 2366 (CS-12 E): sinusrhythmus lagetyp normal unvollständiger rechtsschenkelblock qrs(t) abnorm
    hochlateraler myokardschaden nicht auszuschliessen
19. ECG 2862 (CS-12 E): sinus arrhythmie verdacht auf p-sinistrocardiale lagetyp normal
20. ECG 4189 (CS-12): sinusrhythmus lagetyp normal periphere niederspannung qrs(t) abnormal inferiorer
    myokardschaden nicht auszuschliessen

## General comments

Anything else you noticed:

## SCP code legend

From PTB-XL `scp_statements.csv`.

- ABQRS: abnormal QRS
- AMI: anterior myocardial infarction
- ASMI: anteroseptal myocardial infarction
- CLBBB: (complete) left bundle branch block
- IMI: inferior myocardial infarction
- IRBBB: incomplete right bundle branch block
- ISCAS: ischemic ST-T changes in anteroseptal leads
- ISCIL: ischemic ST-T changes in inferolateral leads
- LMI: lateral myocardial infarction
- LOWT: low amplitude T-waves
- LVH: left ventricular hypertrophy
- NDT: non-diagnostic T abnormalities
- NORM: normal ECG
- SARRH: sinus arrhythmia
- SEHYP: septal hypertrophy
- SR: sinus rhythm
- STD_: non-specific ST depression
- VCLVH: voltage criteria (QRS) for left ventricular hypertrophy

The source data for this sheet are `outputs/experiment024_embedding_geometry_v1/medoids.csv` and
`label_audit.csv`; the results are in [experiment-024-embedding-geometry-results.md](experiment-024-embedding-geometry-results.md).
