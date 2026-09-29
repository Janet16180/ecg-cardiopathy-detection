# Preparing for the cardiologist meeting: ECG screening of university students

29 September 2026. The reader is the project engineer. This guide prepares them for a meeting with a
cardiologist to agree the clinical parameters of a student screening project. A nurse will record ECGs of
university students for free, and a model built here would flag which students need follow-up. Reading time
is about 35 minutes. Sections 1-3 give background, section 4 lists the questions, and section 5 is a pilot
proposal to bring. Project numbers come from the results reports linked in each section. Outside facts come
from the sources in section 7. Anything marked "unverified" is from memory and was not checked against a
fetched source; confirm it before relying on it.

## Contents

1. [ECG in 10 minutes for an engineer](#1-ecg-in-10-minutes-for-an-engineer)
2. [Screening young people: what is different](#2-screening-young-people-what-is-different)
3. [What our model can and cannot do today](#3-what-our-model-can-and-cannot-do-today)
4. [Questions to ask the cardiologist](#4-questions-to-ask-the-cardiologist)
5. [A proposed pilot to bring to the meeting](#5-a-proposed-pilot-to-bring-to-the-meeting)
6. [Glossary](#6-glossary)
7. [References](#7-references)

## 1. ECG in 10 minutes for an engineer

### The signal source

The heart is an electrically excitable muscle. Each beat, a wave of depolarization starts in the sinus node
(a natural pacemaker in the right atrium). It spreads across both atria, pauses in the atrioventricular (AV)
node, runs down fast conduction wiring (the His bundle, the left and right bundle branches and the Purkinje
fibres), and then spreads through the ventricular muscle. Recovery (repolarization) follows.

At the body surface, the sum of all these cell currents looks roughly like one electric dipole whose
direction and size change over the beat. A useful engineering model is a 3D vector `h(t)`, the heart vector.
Each ECG lead measures approximately its dot product with a fixed lead vector `l`: `v(t) ≈ l · h(t)`. The
model is approximate because the torso is not a uniform conductor, but it explains most of what follows.

### Twelve leads are twelve projections

A standard 12-lead ECG uses 10 electrodes: four on the limbs (right arm, left arm, left leg, and right leg as
ground/reference) and six on the chest (V1-V6). The 12 "leads" are voltage differences computed from them:

| Leads | How they are formed | Plane | View |
| --- | --- | --- | --- |
| I, II, III | Differences between limb electrodes (I = LA − RA, II = LL − RA, III = LL − LA) | Frontal (up-down, left-right) | I, aVL: lateral (left side); II, III, aVF: inferior (bottom); aVR: looks from the right shoulder |
| aVR, aVL, aVF | One limb electrode against the mean of the other two | Frontal | Same as above |
| V1-V6 | Each chest electrode against Wilson's central terminal (mean of RA, LA, LL) | Horizontal (front-back, left-right) | V1-V2: septal/right; V3-V4: anterior; V5-V6: lateral |

Only 8 of the 12 are linearly independent: III = II − I, and the three augmented leads are fixed
combinations of I and II. That is why ECG-JEPA uses 8 leads (I, II, V1-V6) with no loss. Cardiologists still
read all 12, because the redundant leads show the same vector from angles that are easier to read by eye.

"Contiguous leads" means neighbours in this geometry, for example II, III and aVF (inferior), or V4-V6. Many
clinical rules require a finding in two contiguous leads. It must appear in two views of the same region,
which guards against a single noisy channel.

### Paper and units

By convention the trace is drawn at 25 mm/s and 10 mm/mV. One small square is 40 ms by 0.1 mV and one large
square is 200 ms by 0.5 mV. Clinical thresholds are often quoted in millimetres: "1 mm of ST depression"
means 0.1 mV. The project data are digital samples at 100-500 Hz, 10 s long. Our models see the same
information as the paper, without the grid.

### Waves and intervals

One beat, in time order:

| Feature | What it is physically | Typical adult values (approximate) |
| --- | --- | --- |
| P wave | Atrial depolarization | Under about 120 ms; small (a few tenths of a mV) |
| PR interval | Start of P to start of QRS: mostly the deliberate AV-node delay | 120-200 ms |
| QRS complex | Ventricular depolarization. It is the largest deflection because the ventricles are the largest mass | Under 120 ms. Q is the first negative dip, R the first positive peak, S the negative after R |
| ST segment | Plateau between depolarization and repolarization. It should sit near the baseline | Near 0 mV. Shifts are measured at the J point (end of QRS) |
| T wave | Ventricular repolarization | Usually the same sign as the QRS in most leads |
| QT interval | Start of QRS to end of T: total ventricular electrical activity | Depends on heart rate, so it is corrected (QTc) |
| U wave | Small wave after T, of debated origin | Often absent |

QT shortens as heart rate rises, so it is normalized. Bazett's correction is `QTc = QT / sqrt(RR)`, with RR
the beat-to-beat interval in seconds. It is the most used, and it is what the athlete criteria assume. It
overcorrects at fast rates and undercorrects at slow ones. The international athlete criteria therefore
suggest repeating the ECG after mild activity if heart rate is below 50 bpm, or after more rest if above 100
bpm, when QTc is borderline ([Drezner 2017, Table 1 note](#ref-drezner)). Computer-measured QTc is about
90-95% accurate, and the criteria recommend manual confirmation ([Drezner 2017](#ref-drezner)).

### Rate, rhythm and axis

- Heart rate is 60/RR. Under 60 bpm is bradycardia and over 100 bpm is tachycardia, by the adult convention.
  Trained young people often rest below 60.
- Sinus rhythm means every QRS follows a normal-looking P wave from the sinus node. Sinus arrhythmia is the
  normal speeding and slowing with breathing (heart rate follows respiration, like a slow FM modulation).
- Arrhythmias are rhythm problems. Examples: atrial fibrillation (chaotic atria and irregular QRS timing),
  flutter, supraventricular tachycardia, premature beats (PVCs from the ventricles, PACs from the atria), and
  AV blocks (P waves that are delayed or fail to conduct).
- The axis is the direction of the mean QRS vector in the frontal plane, in degrees. Roughly −30° to +90° is
  normal. Left axis deviation is −30° to −90°, and the athlete criteria define right axis deviation as over
  +120° ([Drezner 2017](#ref-drezner)).

### What our four abnormal groups mean physically

Our binary label calls an ECG abnormal if it has any code from four PTB-XL superclasses:

| Group | Meaning | Signal-level picture |
| --- | --- | --- |
| MI, myocardial infarction | Heart muscle killed by a blocked artery | Dead tissue contributes no current. The vector points away from the region, which gives deep or wide Q waves in the leads facing it. An acute infarct also shifts the ST segment (a "current of injury", like a DC offset during part of the beat) |
| STTC, ST/T change | Abnormal repolarization: ST depression or elevation, flat or inverted T waves, long QT | Changes in the second half of the beat. The causes are many: ischaemia, hypertrophy strain, cardiomyopathy, electrolytes, drugs, and normal variants. The group is sensitive but not specific |
| CD, conduction disturbance | Faulty wiring: AV blocks, bundle branch blocks, fascicular blocks, pre-excitation (WPW) | A bundle branch block forces the wave through slow muscle instead of fast wiring, so the QRS gets wider and changes shape, like a pulse spread by a slower channel. AV block lengthens PR or drops beats. WPW is an extra path that bypasses the AV delay: short PR plus a slurred QRS start (delta wave) |
| HYP, hypertrophy | Thickened ventricle or enlarged atrium | Bigger muscle means bigger voltage, but voltage also depends on chest-wall thickness, body build, age and sex. Estimating muscle mass from surface voltage is like estimating transmitter power from received signal without knowing the path loss. ECG voltage criteria correlate poorly with wall thickness on imaging ([Drezner 2017](#ref-drezner)) |

"Normal" in our label means sinus rhythm with no other finding
([challenge-label-mapping.md](challenge-label-mapping.md), currently on branch `eda/ningbo-v1`, not yet on
main). Two consequences matter for students. Rhythm and form statements are ignored, so for example a
PTB-XL ECG with atrial fibrillation but no superclass code is not positive. And several findings that
athlete criteria call normal are positive in our label (section 2.4).

## 2. Screening young people: what is different

### 2.1 Base rates: the disease is rare, the death is rarer

- About 0.3% of apparently healthy young athletes have a cardiac disorder associated with sudden cardiac
  death (SCD). The 2020 ESC guidelines give this figure, supported by several screening studies
  ([Pelliccia 2020, section 3.5](#ref-esc2020)). The English FA screened 11,168 adolescent soccer players
  and found 0.38% ([Malhotra 2018](#ref-malhotra)).
- Hypertrophic cardiomyopathy (HCM) was found on echocardiography in 0.17% of 4,111 adults aged 23-35 from
  the general population, "about 2 of 1000" ([Maron 1995](#ref-maron1995)).
- Long QT syndrome: an ECG study of 44,596 newborns suggested a prevalence close to 1 in 2,000 among white
  infants ([Schwartz 2009](#ref-schwartz)).
- Deaths: in US college (NCAA) athletes over 2003-2013, SCD occurred at 1 per 53,703 athlete-years. Males,
  Black athletes and basketball players had higher risk ([Harmon 2015](#ref-harmon)). The ESC cites a
  generally accepted incidence of about 1 in 50,000 per year for college-aged athletes
  ([Pelliccia 2020](#ref-esc2020)).
- Most students are not competitive athletes. In the UK, 7,764 non-athletes aged 14-35 were screened with
  ECG. By the 2010 ESC criteria, 21.8% had "group 2" (potentially pathological) patterns, mostly QTc
  findings. The authors concluded that most were non-specific ([Chandra 2014](#ref-chandra)). There is little
  published data on modern (2017) criteria applied to non-athletes. I did not find a validation study.

The practical consequence: even a good test will produce many more false alarms than true findings. If 0.3%
have disease and a well-read ECG is abnormal in 2-5%, most abnormal ECGs are false positives. In the UK
cohort below, positive ECGs by the refined criteria (4.3%) outnumbered serious diagnoses (0.3%) about 14 to 1
([Dhutia 2016](#ref-dhutia2016)).

### 2.2 The conditions that matter, and what ECG can see

SCD in the young is mostly caused by inherited or congenital heart disease. After age 35, coronary
atherosclerosis dominates ([Pelliccia 2020](#ref-esc2020)). Autopsy registries give different mixes. In the
NCAA series, the most common finding was a structurally normal heart ("autopsy-negative sudden unexplained
death", 25%), and definite HCM was 8% ([Harmon 2015](#ref-harmon)). In a UK registry of 357 athletes (mean
age 29), sudden arrhythmic death syndrome was 42%, myocardial disease 40% (including ARVC 13% and HCM 6%),
and coronary anomalies 5% ([Finocchiaro 2016](#ref-finocchiaro)). In the FA cohort, 6 of the 8 later SCDs
had normal screening results ([Malhotra 2018](#ref-malhotra)).

| Condition | What it is | Can a resting ECG hint at it? |
| --- | --- | --- |
| Hypertrophic cardiomyopathy (HCM) | Genetically thickened heart muscle | Often. Lateral/inferolateral T-wave inversion, ST depression, pathological Q waves. Voltage alone is rarely the only sign: isolated voltage criteria occur in under 2% of HCM patients ([Drezner 2017](#ref-drezner)) |
| Arrhythmogenic cardiomyopathy (ARVC) | Muscle replaced by fat and fibrous tissue, mainly in the right ventricle | Sometimes. Anterior T-wave inversion beyond V2, epsilon wave, low limb-lead voltage, PVCs |
| Dilated cardiomyopathy, LV non-compaction, myocarditis | Weak or inflamed muscle | Sometimes. LBBB, T-wave inversion, Q waves, arrhythmia |
| Wolff-Parkinson-White (WPW) | Extra conduction path between atria and ventricles | Yes, when the pathway conducts at rest: short PR plus delta wave |
| Long QT syndrome | Ion-channel disease that delays repolarization | Yes, often. Prolonged QTc. Some carriers have a normal QTc (unverified, from memory) |
| Short QT syndrome | Very short repolarization | QTc under 320 ms. This is very rare (0.1% of over 18,000 young Britons), and no adverse events occurred in them over 5.3 years. The criteria investigate it only with other warning signs ([Drezner 2017](#ref-drezner)) |
| Brugada syndrome | Sodium-channel disease | Type 1 pattern (coved ST elevation in V1-V2). It can be intermittent and is unmasked by fever or drugs. High placement of V1-V2 changes it ([Drezner 2017](#ref-drezner)) |
| CPVT (catecholaminergic polymorphic VT) | Calcium-handling disease that triggers arrhythmia under adrenaline | Usually not. The resting ECG is typically normal; an exercise test is needed (unverified, from memory) |
| Coronary artery anomalies | An artery that starts or runs in the wrong place | No. They are "not readily detected by ECG" ([Drezner 2017](#ref-drezner)) |
| Aortopathies (for example Marfan) | Weak aortic wall | No. They are detected by physical examination, family history or imaging ([Drezner 2017](#ref-drezner); [AHA 14 elements](#ref-aafp)) |
| Premature coronary atherosclerosis | Early coronary disease | Rarely at rest ([Drezner 2017](#ref-drezner)) |

Table 2 of the international criteria lists the follow-up test for each abnormal finding
([Drezner 2017](#ref-drezner)). Most start with echocardiography. Lateral T-wave inversion adds cardiac
MRI, an exercise ECG and at least 24 h of ECG monitoring. Pre-excitation and arrhythmias go to exercise
testing and electrophysiology. Type 1 Brugada is referred to an electrophysiologist. The pathway, not just
the ECG, determines the cost of each flag.

### 2.3 The athlete criteria

The current standard is the international consensus of 2017 ([Sharma 2017](#ref-sharma), co-published as
[Drezner 2017](#ref-drezner)), agreed in Seattle in February 2015. Its predecessors:

- 2005: first ESC criteria for an abnormal athlete ECG.
- 2010: ESC recommendations split findings into group 1 (common, training-related) and group 2 (uncommon,
  potentially pathological).
- The "Stanford" criteria followed, then the "Seattle criteria" from a 2012 summit, published in 2013. Both
  were more specific (years unverified, from memory; the Seattle publication year matches a later review).
- 2014: the "refined" criteria added a borderline category and adjusted for Black athletes
  ([Sheikh 2014](#ref-sheikh)).
- 2017: international criteria.

The criteria apply to asymptomatic athletes aged 12-35. They assume regular training, typically at least
4-8 hours a week, and advise "prudent application" at lower activity levels. They may need modification
with symptoms or a family history of inherited disease or premature SCD ([Drezner 2017](#ref-drezner)).
Most students will fall outside the athlete definition. This is the first thing to ask about.

The three categories, with the definitions from Table 1 of [Drezner 2017](#ref-drezner):

| Normal (training-related): no further evaluation if asymptomatic, with no significant family history | Borderline: one alone is fine; two or more may warrant evaluation | Abnormal: further evaluation |
| --- | --- | --- |
| Isolated QRS voltage criteria for LVH (SV1 + RV5/RV6 > 3.5 mV) or RVH (RV1 + SV5/SV6 > 1.1 mV) | Left axis deviation (−30° to −90°) | T-wave inversion ≥ 1 mm in two or more contiguous leads (excluding aVR, III, V1), in anterior (V2-V4), lateral, inferolateral or inferior leads, with the exceptions in the first column |
| Incomplete RBBB (QRS < 120 ms) | Left atrial enlargement | ST depression ≥ 0.5 mm in two or more contiguous leads |
| Early repolarization (J-point or ST elevation, J waves, terminal QRS slurring, inferior/lateral leads) | Right axis deviation (> 120°) | Pathological Q waves: Q/R ≥ 0.25 or ≥ 40 ms in two or more leads (excluding III, aVR) |
| Black athlete repolarization variant: J-point and convex ST elevation followed by T-wave inversion in V1-V4 | Right atrial enlargement (P ≥ 2.5 mm in II, III or aVF) | Complete LBBB |
| Juvenile T-wave pattern: T-wave inversion V1-V3 under age 16 | Complete RBBB | Any QRS ≥ 140 ms |
| Sinus bradycardia ≥ 30 bpm; sinus arrhythmia | | Epsilon wave |
| Ectopic atrial or junctional escape rhythm | | Ventricular pre-excitation (PR < 120 ms, delta wave, QRS ≥ 120 ms) |
| First-degree AV block (PR 200-400 ms) | | Prolonged QTc: ≥ 470 ms (male), ≥ 480 ms (female); ≥ 500 ms is marked |
| Mobitz type I (Wenckebach) second-degree AV block | | Brugada type 1 pattern |
| | | Sinus bradycardia < 30 bpm or pauses ≥ 3 s; PR ≥ 400 ms |
| | | Mobitz type II or third-degree AV block |
| | | ≥ 2 PVCs per 10 s tracing; couplets, triplets, non-sustained VT |
| | | Atrial tachyarrhythmias (SVT, atrial fibrillation, flutter) |

The general-medicine QTc limits (over 450 ms in men, over 460 ms in women) are lower. The athlete limits
were raised to about the 99th percentile to limit false positives ([Drezner 2017](#ref-drezner)).

### 2.4 Normal variants that older-adult rules misread

Young, fit people have larger hearts and more vagal (resting) tone. Older-adult rules read several of the
results as disease ([Drezner 2017](#ref-drezner)):

- Voltage criteria for LVH are met by up to 64% of trained athletes (Sokolow-Lyon). Younger age, male sex
  and training all raise voltage.
- Early repolarization is present in up to 45% of white athletes and 63-91% of Black athletes. Over two
  thirds of Black athletes show ST elevation, and up to 25% show T-wave inversion.
- Anterior T-wave inversion (V1-V4) after domed ST elevation is normal in Black athletes. It was seen in 13%
  of 904 Black male athletes, against 4% of Black sedentary controls, with no disease in 5 years of
  follow-up. Extension into V5 is abnormal ([Petek 2024](#ref-pitfalls)).
- The juvenile T-wave pattern (inversion V1-V3) is common at age 12 and rare after 16. It is mostly
  irrelevant for university students, except for younger first-years.
- Sinus bradycardia, first-degree AV block, Wenckebach, and incomplete RBBB come from training.

These variants show up in three places in the project:

| Finding | Athlete criteria | Our current label |
| --- | --- | --- |
| Incomplete RBBB | Normal | Abnormal (PTB-XL `IRBBB` is CD; Challenge code 713426002 is CD). It is audit case 1 in the [clinician review sheet](experiment-024-clinician-review.md) |
| First-degree AV block | Normal (PR 200-400 ms) | Abnormal (CD) |
| Complete RBBB | Borderline | Abnormal (CD) |
| Left or right atrial enlargement | Borderline | Abnormal (HYP) |
| Axis deviation | Borderline | Ignored in PTB-XL; undefined in the Challenge mapping |
| Isolated LVH voltage | Normal | PTB-XL `VCLVH` is a form statement (ignored). Ningbo "left ventricular high voltage" is left undefined (1,874 Ningbo ECGs have it as the only finding; their median age is 68). A reader's diagnosis of LVH is HYP |
| Early repolarization | Normal | Undefined in the Challenge mapping |
| Anterior T-wave inversion in Black athletes | Normal | Likely STTC (abnormal) |
| ≥ 2 PVCs, atrial fibrillation or flutter, SVT | Abnormal | Rhythm statements: ignored or undefined, never positive |
| Brugada type 1, epsilon wave, short QT | Abnormal (short QT only with other signs) | No usable code |
| WPW, long QT, pathological Q, LBBB, ST depression | Abnormal | Abnormal (CD, STTC, MI) |

So the model's target disagrees with the athlete criteria in both directions. It flags several
training-related findings. It is also blind to some rhythm findings the criteria care about, because rhythm
statements never make an ECG positive in our label.

### 2.5 False positives and follow-up cost

Each revision of the criteria cut the share of abnormal ECGs while keeping sensitivity for the detectable
conditions:

| Cohort | 2010 ESC | Seattle | Refined | International |
| --- | ---: | ---: | ---: | ---: |
| 4,297 white elite athletes, UK ([Sheikh 2014](#ref-sheikh)) | 16.2% | 7.1% | 5.3% | – |
| 1,208 Black elite athletes, UK ([Sheikh 2014](#ref-sheikh)) | 40.4% | 18.4% | 11.5% | – |
| 4,925 athletes aged 14-35, UK ([Dhutia 2016](#ref-dhutia2016)) | 21.8% | 6.0% | 4.3% | – |
| Several large studies ([Petek 2024](#ref-pitfalls)) | – | – | – | 1.3-6.8% |

In Sheikh 2014 all three older criteria still identified 98.1% of 103 athletes with HCM. Under the
international criteria, US college athletes had 1.5-2.1% abnormal ECGs. Expert readers applying the criteria
to 5,258 college athletes found 1.3%. Arab and Black athletes in Qatar had 6.8%. Some groups were far higher:
Ghanaian soccer players 23.3% and NBA players 15.6% ([Petek 2024](#ref-pitfalls)).

Costs, from UK NHS tariffs (2011-2014 data):

- With the 2010 ESC criteria, 11.2% of athletes needed an echocardiogram, 1.7% an exercise test, 1.2% a
  Holter monitor, 1.2% cardiac MRI and 0.4% other tests.
- Screening cost $110 per athlete and $35,993 per serious diagnosis.
- The refined criteria brought this to $87 per athlete and $28,510 per diagnosis. 15 athletes (0.3%) had
  potentially serious disease.
- Source: [Dhutia 2016](#ref-dhutia2016).

A US decision model estimated that adding ECG to history and examination costs $42,900 per life-year saved
([Wheeler 2010](#ref-wheeler)). This depends heavily on local prices and effectiveness assumptions.

Reader variability matters as much as the criteria. Eight UK cardiologists each read 400 athlete ECGs:
- Agreement on "abnormal" was only moderate even among experienced readers (kappa 0.40-0.53).
- Inexperienced cardiologists referred far more (odds ratio 4.74).
- The downstream cost was $175 against $101 per athlete.
- Source: [Dhutia 2017](#ref-dhutia2017).

This bears directly on labels. One cardiologist's reading is a noisy label, and a second reader measures how
noisy.

### 2.6 Society positions

- **AHA/ACC (US).** A 2014 statement concluded that the data do not support universal ECG screening of
  people aged 12-25 as a public health measure ([Maron 2014](#ref-maron2014);
  [MDedge summary](#ref-mdedge)). The 2015 AHA/ACC task force recommends a 14-element history and physical
  examination. Universal ECG screening of the young general population is not recommended. However, ECG "may
  be considered as part of screening in smaller cohorts of young (12 to 25 years of age) healthy persons,
  with the physician closely involved and quality control measures in place" ([AAFP summary](#ref-aafp)).
  A university program fits that sentence most closely.
- **ESC (Europe).** Pre-participation screening of athletes is "universally supported" by major societies,
  but the best method for young competitive athletes "remains controversial". With experienced readers and
  contemporary standards, ECG screening "outperforms history and physical examination in all statistical
  measures of performance". Routine screening echocardiography is not recommended
  ([Pelliccia 2020](#ref-esc2020)).
- **Italy** has mandated ECG-inclusive screening of competitive athletes since 1982. In the Veneto region,
  SCD among screened athletes fell by 89%, from 3.6 to 0.4 per 100,000 person-years, while unscreened
  non-athletes saw no significant change ([Corrado 2006](#ref-corrado)). This is observational evidence and
  is still debated.

In short, the US and European positions differ on athletes and mass screening. Both accept careful,
physician-led ECG screening of defined groups, and both insist on a history questionnaire as well.

### 2.7 What the nurse will face

From [Drezner 2017](#ref-drezner) unless marked:

- Precordial placement is the commonest error. V1 and V2 belong in the fourth intercostal space beside the
  sternum. V4 goes in the fifth space at the mid-clavicular line, V3 midway between V2 and V4, and V5-V6
  level with V4 at the anterior and mid-axillary lines. They do not follow the rib curve.
- Placing V1-V2 too high (second or third space) can mimic a Brugada type 2 pattern or ST elevation. Placing
  them too low can create pseudo-Q waves and hide ST depression.
- In one survey, only 49% of nurses and 16% of cardiologists marked V1 correctly, against 90% of cardiac
  technicians ([Rajaganeshan 2008](#ref-rajaganeshan)).
- Placement in women around breast tissue may differ, particularly at large screening events
  ([Petek 2024](#ref-pitfalls)).
- Limb lead reversal: swapping the arm electrodes gives negative P, QRS and T in lead I and aVL but normal
  V5-V6. Swapping any limb cable with the neutral (right leg) cable gives a near-flat lead I, II or III.
  Reversals can mimic infarction, ischaemia or conduction disease, and they can be detected by algorithm
  ([Batchvarov 2007](#ref-batchvarov)).
- Noise and drift come from muscle tremor (cold, nerves, talking), poor skin contact, cable movement and
  mains interference.
- Device filters change the waveform:
  - Diagnostic mode typically uses a high-pass cutoff of about 0.05 Hz and a low-pass of about 150 Hz.
  - "Monitoring" or "muscle" filters (for example 0.5-40 Hz) distort the ST segment and reduce QRS
    amplitude.
  - These values are unverified, from memory; see [Kligfield 2007](#ref-kligfield) for the standard.
  - The project already sees this problem: SPH is far more heavily band-pass filtered than PTB-XL, and
    thresholds did not transfer between them (section 3).
- Position: standard ECGs are recorded supine at rest. Heart rate, QT and some amplitudes change with posture
  and anxiety (unverified, from memory). Placing the limb electrodes on the torso instead of the limbs also
  changes the waveform (unverified, from memory).

## 3. What our model can and cannot do today

### 3.1 What it is

The baseline is a frozen, publicly released pretrained encoder (xECG or ECG-JEPA) with a simple logistic
readout, applied to a 10 s 12-lead ECG. The readout is trained to predict the binary label of section 1:
abnormal means any MI, STTC, CD or HYP code; normal means sinus rhythm with no other finding
([findings](findings-2026-09-28.md)). The labels are ECG annotations by human readers (PTB-XL, SPH and the
Challenge sources). MIMIC machine labels are not used.

### 3.2 The numbers

| Question | Result | Source |
| --- | --- | --- |
| Ranking abnormal above normal, PTB-XL development | AUROC 0.959 (JEPA), 0.962 (xECG) with all 15,359 labels | [025](experiment-025-label-efficiency-results.md) |
| Same with only 100 labels | 0.922, 0.921 | [025](experiment-025-label-efficiency-results.md) |
| A new hospital (SPH, China), no retraining | AUROC 0.911, 0.915 | [022](experiment-022-sph-external-readout-results.md) |
| A threshold set for 95% sensitivity on PTB-XL, applied at SPH | Sensitivity 0.916 / 0.922; specificity 0.663 / 0.646; about 34-35% of normal ECGs flagged | [027](experiment-027-calibrated-threshold-results.md) |
| Same threshold at an assumed 1% prevalence | About 343-360 referrals per 1,000 people, PPV about 0.03 | [027](experiment-027-calibrated-threshold-results.md) |
| Distance from normal only (no abnormal labels) | AUROC 0.923 (xECG) on PTB-XL; 0.858 at SPH | [026](experiment-026-normal-manifold-results.md) |
| Echo-confirmed structural disease (EchoNext) with a head trained on echo labels | AUROC 0.838 (xECG) | [023](experiment-023-echonext-readout-results.md) |
| Same disease with the ECG-abnormality head, unchanged | About 0.75 | [023](experiment-023-echonext-readout-results.md) |

### 3.3 Its limits for students

1. **It learned mostly from older hospital patients.** Only about 2,800 labeled ECGs are from people aged
   18-25, and about 7,800 from ages 18-35. Most are from people over 50. PTB-XL aged 18-35 has 1,575
   normal-only and 414 abnormal ECGs. SPH, the external test set, has 5,399 ECGs aged 18-35. No experiment
   has yet reported performance in the young subgroup. That is a cheap analysis to run before the pilot.
2. **It detects ECG annotations, not disease.** The target is "a hospital reader wrote an abnormal code".
   Against echo-confirmed structural disease, this head reaches only about 0.75 AUROC (023). A head trained
   on echo labels does better (0.838). Claims of "detecting cardiopathy" need disease-confirmed labels.
3. **Its target is not the athlete criteria.** It flags incomplete RBBB, first-degree AV block and atrial
   enlargement, which are normal or borderline in young athletes. It ignores PVCs, atrial arrhythmias and
   Brugada patterns, which are abnormal (section 2.4). It does not measure QTc to the criteria's standard;
   that needs a measurement step, preferably confirmed by a human.
4. **Thresholds do not transfer.** The ranking held at a new hospital, but the operating point did not.
   Sensitivity fell from 95% to about 92%, and probabilities were too high on average (calibration
   intercept −1.1 to −1.9) (027). Any threshold must be set on local ECGs.
5. **Its current operating point refers far too many people.** About 34% of normal ECGs are flagged, against
   1.3-6.8% abnormal ECGs for expert readers using the international criteria (section 2.5). This point
   was chosen for 95% sensitivity on the broad label. The sensitivity at a 2-5% referral rate has not been
   measured yet. It needs only the saved 022 predictions.
6. **"Normal" shifts between sites.** The distance-from-normal detector lost more at SPH (0.923 to 0.858)
   than the classifier did. A normal reference fitted on local student ECGs is the obvious fix, and it is
   untested (backlog `local_normal_manifold`).
7. **Devices leave a fingerprint.** Unsupervised clusters in the encoders followed recording device and heart
   rate more than diagnosis ([024](experiment-024-embedding-geometry-results.md)). A new cart at the
   university is a new domain.
8. **One run each.** Each result is one fit and one seed. PTB-XL development patients have been examined
   many times.

What the model is plausibly good for now is triage support: ranking ECGs so the cardiologist reads the most
suspicious first, or a second opinion, running silently next to the cardiologist during a pilot. It is not
ready to be the referral decision.

## 4. Questions to ask the cardiologist

### 4.1 The ten to ask first

If time is short, ask these:

1. For students, what counts as abnormal? Would you use the 2017 international criteria as written, even
   for students who are not athletes?
2. Which normal variants should the model never flag (incomplete RBBB, first-degree AV block, early
   repolarization, isolated voltage, and so on)?
3. Out of 1,000 students, how many referrals can your service absorb? How many misses of a serious
   condition would you accept?
4. What exactly happens after a flag: repeat ECG, echo, Holter, referral, with what waiting times and costs?
5. Which device will the nurse use, and can it export the raw digital waveform with filter settings?
6. Would you read and label a pilot set? How many ECGs, how fast, and on what form?
7. Should a second reader read some of them, and who?
8. Which questionnaire items should go with each ECG (symptoms, family history, athlete status,
   medications)?
9. Who is clinically responsible for a result, and what do we tell a student with an abnormal ECG?
10. Can you supply ECGs of young patients with confirmed conditions, so sensitivity can be checked?

### 4.2 All questions, grouped by importance

Each question has one line on why it matters for the model.

**A. The referral rule.**

1. Would you use the 2017 international criteria for all students, or only for athletes? What would you
   change for sedentary students? *The rule defines the label; without it we are training on the wrong
   target.*
2. Do you apply the borderline rule (one finding is fine, two or more are evaluated)? *It is an explicit
   counting rule that can be encoded.*
3. Would you use the athlete QTc limits (470 male, 480 female) or the general ones (450, 460)? *QTc findings
   were the largest group of "abnormal" ECGs in young non-athletes ([Chandra 2014](#ref-chandra)); this one
   choice moves the referral rate.*
4. Is the output normal/abnormal, or normal/borderline/abnormal with a named finding? *Named findings let us
   train and audit per finding instead of one opaque score.*
5. Should the model also target echo-confirmed disease, or only the ECG criteria? *023 showed these are
   different targets that need different labels.*
6. Are there findings the criteria miss that you want flagged anyway? *Anything outside the criteria must be
   stated now, or it will not be in the label.*

**B. Normal variants to ignore.**

7. Please confirm the variants to treat as normal in students: incomplete RBBB, first-degree AV block,
   Wenckebach, sinus bradycardia (how low?), early repolarization, isolated LVH/RVH voltage. *Our current
   label marks several of these as abnormal, so the readout must be retrained or remapped.*
8. How should we handle ethnicity-dependent patterns such as anterior T-wave inversion in Black students? Do
   we need to record ethnicity? *The criteria depend on ethnicity; if we do not record it, the model cannot
   apply the rule.*
9. Are there variants common in the local population that the international criteria do not cover? *The
   criteria were built mainly from European and US cohorts, and abnormal rates reached 20% or more in some
   populations ([Petek 2024](#ref-pitfalls)).*
10. Should first-year students under 16 or 17 be handled differently (juvenile pattern)? *It is a small
    age-dependent rule, easy to encode if the age is known.*

**C. False alarms versus misses.**

11. How many referrals per 1,000 students are acceptable? *This, not AUROC, sets the threshold. Our current
    point refers about 340 per 1,000 (027).*
12. Which conditions must never be missed, and which are acceptable to miss at a first screen? *We can set
    per-finding sensitivity targets instead of one global one.*
13. Is a two-stage design acceptable, with the model filtering and you reading everything flagged plus a
    random sample of the rest? *This keeps your workload bounded and still measures what the model misses.*
14. How would you like uncertainty communicated: a score, a category, or "please read this one"? *It
    determines the output format and whether we calibrate probabilities.*

**D. The follow-up pathway after a flag.**

15. What happens after a flag, step by step: repeat ECG, echo, Holter, exercise test, MRI, genetics, referral?
    *The cost and harm of a false alarm depend on this pathway, so it sets how many false alarms are
    tolerable.*
16. What are the capacity and waiting time for echo and cardiology clinic, and who pays? *Capacity bounds the
    referral rate we can allow.*
17. What is done with a technically poor ECG: repeat on the spot, or recall? *It defines a "reject, repeat"
    output separate from "abnormal".*
18. Will follow-up results (echo findings, final diagnosis) come back to the project? *Outcomes are the only
    route to disease-confirmed labels and to measuring real sensitivity.*

**E. The recording protocol.**

19. Which device model and software version? Can it export raw digital samples (sampling rate, resolution,
    units) and not just a PDF or image? *The model needs the raw signal, and every new device is a domain
    shift.*
20. Which filters will be on (high-pass, low-pass, mains, muscle filter)? Can we fix them in diagnostic mode
    for everyone? *Filters change ST and QRS shape; SPH's heavy filtering is one reason thresholds did not
    transfer.*
21. Supine, how many minutes of rest, and which room conditions? *Heart rate and QTc depend on rest and
    anxiety, and the criteria assume a resting ECG.*
22. How will the nurse be trained and checked on electrode placement, especially V1-V2 and in women? Should
    each ECG have a quick visual check before the student leaves? *Misplacement creates false Brugada, Q-wave
    and ST findings; catching it on the day avoids a recall.*
23. Should the device's automatic interpretation be saved? *It is a free baseline to compare against, but it
    must never be used as a label.*
24. Will the cart's own measurements (heart rate, PR, QRS, QT, QTc, axis) be exported? *They feed a
    rule-based layer for interval criteria that the encoder does not measure explicitly.*

**F. Labeling a pilot set.**

25. Could you read a pilot set of a few hundred to a thousand ECGs? How long does one take you? *Reader time
    is the main cost of the pilot and sets N.*
26. Would you use a structured form: category, named findings, quality, "would you refer?", confidence?
    *Structured labels are machine-usable; free text is not.*
27. Would you first review our existing [clinician review sheet](experiment-024-clinician-review.md) (30-60
    minutes)? *It calibrates how your reading compares with PTB-XL's labels, and several cases (for example
    incomplete RBBB) are exactly the student questions.*
28. Can you provide ECGs of young people with confirmed conditions (HCM, WPW, long QT, and so on) from your
    clinic, with consent? *A student pilot will contain too few positives to measure sensitivity (section 5).*
29. Should we also label a sample of older public ECGs aged 18-35 (PTB-XL, SPH) with the student criteria?
    *It would give a young, criteria-matched training and test set without new recordings.*

**G. A second reader.**

30. Should a second cardiologist read all flagged ECGs and a random sample of normal ones? *Agreement among
    experienced readers is only moderate (kappa 0.40-0.53) ([Dhutia 2017](#ref-dhutia2017)); without a second
    reader we cannot tell model error from label noise.*
31. How should disagreements be resolved: consensus, a third reader, or keeping both labels? *This decides
    whether the label is one value or a distribution.*

**H. What else to record.**

32. Which history items should be collected? The AHA 14-element list covers exertional chest pain,
    unexplained syncope, exertional breathlessness or palpitations, a known murmur, high blood pressure,
    previous restriction or cardiac tests, family history of premature or unexplained death, heart
    disability or inherited heart disease, and examination findings ([AAFP summary](#ref-aafp)). *The
    athlete criteria are only valid for asymptomatic people with no concerning family history. The
    questionnaire changes how the ECG is read, and it is a model input.*
33. Athlete status, sport, and training hours per week? *The criteria assume at least 4-8 hours a week, and
    training explains many normal variants.*
34. Medications and substances (QT-prolonging drugs, stimulants, caffeine), recent illness or fever? *They
    change QT and can unmask Brugada; without them, some findings are uninterpretable.*
35. Age, sex, height, weight, ethnicity? *Voltage, QTc limits and variants depend on them. Age and sex alone
    already carry some signal in our data.*
36. Should a physical examination (blood pressure, murmur) be part of the visit? *It catches conditions ECG
    cannot see, such as aortopathy and valve disease.*

**I. Ethics, consent and data.**

37. Is this research, clinical service, or both? Who is the responsible clinician for each result? *It
    determines approvals and whether the model may influence care during the pilot.*
38. What will a student be told before consenting? This should cover the chance of a false alarm, that a
    normal ECG does not rule everything out, and what happens after a positive. *Informed consent must state
    these limits honestly (section 2.2).*
39. Can the data be kept for research, with what de-identification, where, and for how long? Can students
    opt out of research use but still get their ECG read? *Data rights decide what we can train on.*
40. Is there a local ethics committee and data protection officer to involve now? *Approval takes time and
    should start before recording.*

### 4.3 Ethics and regulation in general terms

The country is not known, so these are general points, not legal advice.

- **Screening principles.** The WHO principles of [Wilson and Jungner (1968)](#ref-who) are the usual
  checklist. The condition should be important. There should be an accepted treatment, facilities for
  diagnosis and treatment, a suitable and acceptable test, an agreed policy on whom to treat, and costs that
  are justified. Screening should be a continuing process. (The list is summarized from memory; the WHO
  monograph is linked.) A screening program is only as good as its follow-up pathway.
- **Consent.** Students should understand that the ECG is voluntary. They should know the chance and
  consequence of a false alarm (possibly further tests and temporary sports restriction), that a normal ECG
  does not exclude coronary anomalies or intermittent conditions, and how their data will be used.
- **After a positive.** There must be a named clinician and a defined pathway before the first recording.
  Nobody should get an "abnormal" result with nowhere to go.
- **Research tool versus diagnostic device.** Software intended for a medical purpose is regulated as a
  medical device in many jurisdictions. The IMDRF definition used by the FDA is "software intended to be
  used for one or more medical purposes that perform these purposes without being part of a hardware
  medical device" ([FDA SaMD](#ref-fda)). In a pilot, the model should run silently: the cardiologist's
  reading drives care, and the model's output is recorded but not acted on. Using it to decide referrals
  would likely require regulatory clearance (general point, unverified for any specific country).
- **Data protection.** ECGs linked to identity are health data. They are "special category" data under the
  EU GDPR, and similar rules apply elsewhere (from memory, unverified). Keep the key linking IDs to
  identities separate from the waveforms. Collect only what is needed. Agree retention and access in
  advance. This project already keeps training data local and out of Git.

## 5. A proposed pilot to bring to the meeting

### Design

1. **Before recording.** Agree the reading criteria (section 4.2 A-B), a structured reading form, the
   questionnaire, the device and its fixed settings, and the follow-up pathway. Start ethics and data
   protection approval.
2. **Feasibility, about 50-100 students.** Check the raw export, the signal quality and the reading time. Fix
   the protocol.
3. **Main pilot, N students (provisional, about 1,000).**
   - The nurse records a supine resting 12-lead ECG plus the questionnaire.
   - The cardiologist reads every ECG with the agreed criteria.
   - A second reader reads all abnormal or borderline ECGs and a random 10-20% of normal ones.
   - The model runs silently and does not affect care.
   - Follow-up outcomes are recorded where available.
4. **Uses of the data.**
   - Local calibration: refit the probability calibration and choose the threshold on student ECGs, as 027
     showed is necessary.
   - A local normal reference: fit the distance-from-normal detector (026) on normal student ECGs.
   - Measure the false-alarm rate and inter-reader agreement.
   - Train and test a readout for the agreed student label.
5. **Sensitivity from elsewhere.** Measure sensitivity on enriched positives: young patients from the
   cardiologist's clinic, and PTB-XL or SPH ECGs aged 18-35 relabeled with the student criteria.

### Why about 1,000, and why it is provisional

- **Negatives are plentiful and set the false-alarm rate.**
  - If 2-5% of students read abnormal, as in athlete studies, N = 1,000 gives about 950-980 normal ECGs.
  - That estimates a 5% referral rate to about ±1.4 points, or a specificity near 0.65 to about ±3 points
    (95% intervals).
  - With 500 students the widths are about ±1.9 and ±4.2 points.
- **Local data fix thresholds, not rankings.** In 027, the ranking transferred to SPH but the threshold and
  calibration did not.
  - The calibration set there had 348 positives, and the threshold was still imprecise.
  - The backlog item `site_recalibration` will measure how many local ECGs (50-1,000) restore the operating
    point, by simulation on SPH.
  - Until it runs, N is an estimate, not a result.
- **A normal-only model needs normals, not positives.** 026 fitted its normal reference on 5,872 PTB-XL
  normal ECGs, projected to 64 components; a full covariance in 64 dimensions has 2,080 free parameters.
  About 1,000 local normals seems a reasonable target, but `local_normal_manifold` (50-1,000 local normals,
  simulated on SPH) has not run.
- **Labels beyond a few thousand add little for this target.**
  - On PTB-XL, the frozen encoders reached 0.92 AUROC with 100 labels and 0.947 with 1,000, and the curve
    was nearly flat above 4,000 (025).
  - For echo-confirmed disease, 250 labels beat the cart's measurements, and near-ceiling needed about 4,000
    (028).
  - So a pilot of about 1,000 is mainly for calibration and a local reference. It is not for training a
    model from scratch.
- **Positives are too few to measure sensitivity locally.**
  - At 1-5% prevalence, 1,000 students give only about 10-50 abnormal ECGs, and only a handful with serious
    disease (about 0.3%, section 2.1).
  - With 20 positives, 19 detected gives a 95% interval of 75-99.9%.
  - A ±5-point estimate around 95% needs about 73 positives: about 3,800 students at 2% prevalence, or 7,600
    at 1%.
  - Sensitivity must therefore come from enriched sets (step 5).
- **Reader time is the real constraint.** Ask how long one reading takes. N should be set by the
  cardiologist's available hours as much as by statistics.

### What not to promise

- The pilot will not show that the model detects heart disease in students. It measures agreement with a
  cardiologist's ECG reading and the false-alarm rate.
- Disease-level claims need follow-up outcomes or echo labels. 023 showed that the ECG-abnormality head and
  structural disease are different targets.
- The pilot cannot validate referral decisions. The project label is an annotation proxy, and it does not
  establish that a student is healthy.

## 6. Glossary

- **ARVC / arrhythmogenic cardiomyopathy.** Inherited disease in which heart muscle is replaced by fat and
  scar, mainly in the right ventricle. It causes arrhythmias, often during exercise.
- **Atrial fibrillation (AF) / flutter.** Chaotic (AF) or circular (flutter) electrical activity in the
  atria. In AF the QRS timing is irregular.
- **AV block (first, second, third degree).** Delay or failure of conduction through the AV node. First
  degree: long PR. Mobitz I (Wenckebach): PR lengthens until a beat drops; often benign in the young. Mobitz
  II: sudden dropped beats with a fixed PR; abnormal. Third degree: no conduction at all.
- **Axis.** Direction of the mean QRS vector in the frontal plane.
- **Borderline finding.** In the athlete criteria, a finding that is fine alone but warrants evaluation if
  two or more are present.
- **Brugada pattern / syndrome.** Coved ST elevation in V1-V2 (type 1) from a sodium-channel disorder,
  linked to sudden death. Type 2 is a "saddleback" pattern and less specific.
- **Bundle branch block (RBBB, LBBB).** Blocked conduction in the right or left branch, which widens the QRS
  (≥ 120 ms when complete). Incomplete RBBB is a narrower version, normal in athletes.
- **Calibration.** Whether predicted probabilities match observed frequencies. The calibration intercept and
  slope measure the mismatch.
- **CPVT.** Catecholaminergic polymorphic ventricular tachycardia: an inherited arrhythmia triggered by
  adrenaline, usually invisible on a resting ECG.
- **Delta wave / pre-excitation / WPW.** An extra pathway conducts early, giving a short PR and a slurred QRS
  start. WPW syndrome is the pattern plus arrhythmias.
- **Early repolarization.** J-point elevation with ST elevation or notching, common and usually benign in
  young and athletic people.
- **Echocardiogram (echo).** Ultrasound of the heart. It shows structure (wall thickness, valves) and pumping
  (ejection fraction). The usual first test after an abnormal ECG.
- **Ejection fraction (LVEF).** Fraction of blood the left ventricle pumps out per beat. Normal is about 55%
  or more; 45% or below was one EchoNext component.
- **Epsilon wave.** A small notch after the QRS in V1-V3, a sign of ARVC.
- **HCM.** Hypertrophic cardiomyopathy: inherited thickening of the heart muscle; a leading cause of SCD in
  the young.
- **Holter.** Portable ECG recorder worn for 24 h or longer to capture intermittent arrhythmias.
- **J point.** The junction between the end of the QRS and the start of the ST segment.
- **Juvenile T-wave pattern.** T-wave inversion in V1-V3 before about age 16, normal in adolescents.
- **LVH / RVH.** Left or right ventricular hypertrophy. On ECG, suggested by large QRS voltages, but voltage
  alone is weak evidence, especially in the young.
- **Long QT / short QT.** Abnormally long or short QTc, often from inherited ion-channel disease; risk of
  dangerous arrhythmia.
- **Myocarditis.** Inflammation of the heart muscle, often after a viral infection.
- **NPV / PPV.** Negative and positive predictive value: the chance that a negative (or positive) result is
  correct. Both depend strongly on prevalence.
- **PVC / PAC.** Premature ventricular or atrial contraction: an early extra beat.
- **Q wave (pathological).** A deep or wide initial negative QRS deflection, suggesting old infarction or
  cardiomyopathy.
- **QTc.** QT corrected for heart rate, usually by Bazett's formula.
- **SADS / autopsy-negative sudden death.** Sudden death with a structurally normal heart at autopsy,
  presumed arrhythmic.
- **SCD.** Sudden cardiac death.
- **Sensitivity / specificity.** Fraction of true positives flagged, and of true negatives not flagged.
- **Sinus rhythm / bradycardia / tachycardia / arrhythmia.** Normal pacemaker rhythm; slow (< 60 bpm);
  fast (> 100 bpm); varying with breathing.
- **ST depression / elevation.** ST segment below or above baseline, measured in mm (1 mm = 0.1 mV).
- **T-wave inversion (TWI).** A negative T wave where it is normally positive; significant at ≥ 1 mm in two
  contiguous leads by the athlete criteria.
- **Voltage criteria (Sokolow-Lyon).** SV1 + RV5 or RV6 > 3.5 mV for LVH.

## 7. References

Fetched and read for this guide on 29 September 2026, except where marked.

<a id="ref-sharma"></a>
- Sharma S, Drezner JA, Baggish A, et al. International Recommendations for Electrocardiographic
  Interpretation in Athletes. J Am Coll Cardiol 2017;69:1057-1075.
  [doi:10.1016/j.jacc.2017.01.015](https://doi.org/10.1016/j.jacc.2017.01.015). The abstract was read; the
  full text was not accessible.

<a id="ref-drezner"></a>
- Drezner JA, Sharma S, Baggish A, et al. International criteria for electrocardiographic interpretation in
  athletes: consensus statement. Br J Sports Med 2017;51:704-731.
  [doi:10.1136/bjsports-2016-097331](https://doi.org/10.1136/bjsports-2016-097331). Full text read from the
  [University of Washington copy](https://uwsportscardiology.org/wp-content/uploads/sites/2/2018/04/International-Criteria-for-ECG-Interpretation_BJSM-2017.pdf).
  Summary diagram:
  [UW Sports Cardiology](https://uwsportscardiology.org/wp-content/themes/flatsome-child/inc/modules/1/story_content/external_files/International%20Criteria%20for%20ECG%20Interpretation%20in%20Athletes_Diagram.pdf).

<a id="ref-pitfalls"></a>
- Petek BJ, Drezner JA, Churchill TW. The International Criteria for Electrocardiogram Interpretation in
  Athletes: Common Pitfalls and Future Directions. Card Electrophysiol Clin 2024.
  [PMC11207195](https://pmc.ncbi.nlm.nih.gov/articles/PMC11207195/).

<a id="ref-sheikh"></a>
- Sheikh N, Papadakis M, Ghani S, et al. Comparison of electrocardiographic criteria for the detection of
  cardiac abnormalities in elite black and white athletes. Circulation 2014;129:1637-1649.
  [doi:10.1161/CIRCULATIONAHA.113.006179](https://doi.org/10.1161/CIRCULATIONAHA.113.006179).

<a id="ref-dhutia2016"></a>
- Dhutia H, Malhotra A, Gabus V, et al. Cost implications of using different ECG criteria for screening
  young athletes in the United Kingdom. J Am Coll Cardiol 2016;68:702-711.
  [doi:10.1016/j.jacc.2016.05.076](https://doi.org/10.1016/j.jacc.2016.05.076).

<a id="ref-dhutia2017"></a>
- Dhutia H, Malhotra A, Yeo TJ, et al. Inter-rater reliability and downstream financial implications of
  electrocardiography screening in young athletes. Circ Cardiovasc Qual Outcomes 2017;10:e003306.
  [doi:10.1161/CIRCOUTCOMES.116.003306](https://doi.org/10.1161/CIRCOUTCOMES.116.003306).

<a id="ref-chandra"></a>
- Chandra N, Bastiaenen R, Papadakis M, et al. Prevalence of electrocardiographic anomalies in young
  individuals: relevance to a nationwide cardiac screening program. J Am Coll Cardiol 2014;63:2028-2034.
  [doi:10.1016/j.jacc.2014.01.046](https://doi.org/10.1016/j.jacc.2014.01.046).

<a id="ref-malhotra"></a>
- Malhotra A, Dhutia H, Finocchiaro G, et al. Outcomes of cardiac screening in adolescent soccer players.
  N Engl J Med 2018;379:524-534. [doi:10.1056/NEJMoa1714719](https://doi.org/10.1056/NEJMoa1714719).

<a id="ref-harmon"></a>
- Harmon KG, Asif IM, Maleszewski JJ, et al. Incidence, cause, and comparative frequency of sudden cardiac
  death in National Collegiate Athletic Association athletes: a decade in review. Circulation
  2015;132:10-19. [doi:10.1161/CIRCULATIONAHA.115.015431](https://doi.org/10.1161/CIRCULATIONAHA.115.015431).

<a id="ref-finocchiaro"></a>
- Finocchiaro G, Papadakis M, Robertus JL, et al. Etiology of sudden death in sports: insights from a
  United Kingdom regional registry. J Am Coll Cardiol 2016;67:2108-2115.
  [doi:10.1016/j.jacc.2016.02.062](https://doi.org/10.1016/j.jacc.2016.02.062).

<a id="ref-corrado"></a>
- Corrado D, Basso C, Pavei A, et al. Trends in sudden cardiovascular death in young competitive athletes
  after implementation of a preparticipation screening program. JAMA 2006;296:1593-1601.
  [doi:10.1001/jama.296.13.1593](https://doi.org/10.1001/jama.296.13.1593).

<a id="ref-maron2014"></a>
- Maron BJ, Friedman RA, Kligfield P, et al. Assessment of the 12-lead ECG as a screening test for detection
  of cardiovascular disease in healthy general populations of young people (12-25 years of age): a
  scientific statement from the AHA and ACC. Circulation 2014;130:1303-1334.
  [doi:10.1161/CIR.0000000000000025](https://doi.org/10.1161/CIR.0000000000000025). The citation was
  verified; the full text was not accessible.

<a id="ref-mdedge"></a>
- MDedge. AHA/ACC: No to universal ECG screen in healthy young people (news summary of Maron 2014).
  [Link](https://blogs.the-hospitalist.org/content/aha/acc-no-universal-ecg-screen-healthy-young-people).

<a id="ref-aafp"></a>
- American Family Physician. Preparticipation screening for CVD in competitive athletes: recommendations
  from the AHA/ACC (2016 summary of the 2015 AHA/ACC task force 2).
  [Link](https://www.aafp.org/pubs/afp/issues/2016/0715/p170.html). Original:
  Maron BJ, Levine BD, Washington RL, et al. Circulation 2015;132:e267-e272,
  [doi:10.1161/CIR.0000000000000238](https://doi.org/10.1161/CIR.0000000000000238) (not read).

<a id="ref-esc2020"></a>
- Pelliccia A, Sharma S, Gati S, et al. 2020 ESC Guidelines on sports cardiology and exercise in patients
  with cardiovascular disease. Eur Heart J 2021;42:17-96.
  [Link](https://academic.oup.com/eurheartj/article/42/1/17/5898937). Sections 3.1-3.7 were read from a
  hosted copy.

<a id="ref-wheeler"></a>
- Wheeler MT, Heidenreich PA, Froelicher VF, et al. Cost-effectiveness of preparticipation screening for
  prevention of sudden cardiac death in young athletes. Ann Intern Med 2010;152:276-286.
  [doi:10.7326/0003-4819-152-5-201003020-00005](https://doi.org/10.7326/0003-4819-152-5-201003020-00005).
  The abstract was read.

<a id="ref-maron1995"></a>
- Maron BJ, Gardin JM, Flack JM, et al. Prevalence of hypertrophic cardiomyopathy in a general population
  of young adults (CARDIA). Circulation 1995;92:785-789.
  [doi:10.1161/01.cir.92.4.785](https://doi.org/10.1161/01.cir.92.4.785).

<a id="ref-schwartz"></a>
- Schwartz PJ, Stramba-Badiale M, Crotti L, et al. Prevalence of the congenital long-QT syndrome.
  Circulation 2009;120:1761-1767.
  [doi:10.1161/CIRCULATIONAHA.109.863209](https://doi.org/10.1161/CIRCULATIONAHA.109.863209).

<a id="ref-batchvarov"></a>
- Batchvarov VN, Malik M, Camm AJ. Incorrect electrode cable connection during electrocardiographic
  recording. Europace 2007;9:1081-1090.
  [doi:10.1093/europace/eum198](https://doi.org/10.1093/europace/eum198).

<a id="ref-rajaganeshan"></a>
- Rajaganeshan R, Ludlam CL, Francis DP, et al. Accuracy in ECG lead placement among technicians, nurses,
  general physicians and cardiologists. Int J Clin Pract 2008;62:65-70.
  [PubMed 17764456](https://pubmed.ncbi.nlm.nih.gov/17764456/).

<a id="ref-kligfield"></a>
- Kligfield P, Gettes LS, Bailey JJ, et al. Recommendations for the standardization and interpretation of the
  electrocardiogram, part I: the electrocardiogram and its technology. Circulation 2007;115:1306-1324.
  [Link](https://www.ahajournals.org/doi/10.1161/circulationaha.106.180200). Not read; the filter values in
  section 2.7 are from memory.

<a id="ref-who"></a>
- Wilson JMG, Jungner G. Principles and practice of screening for disease. WHO Public Health Papers 34, 1968.
  [WHO IRIS](https://apps.who.int/iris/handle/10665/37650). Not read; the principles are summarized from
  memory.

<a id="ref-fda"></a>
- US FDA. Software as a Medical Device (SaMD).
  [Link](https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd).

Project documents: [findings of 28 September](findings-2026-09-28.md),
[papers versus results](reflection-papers-vs-results-2026-09-29.md),
[literature review](literature-review-2026-09-28.md), results of Experiments
[022](experiment-022-sph-external-readout-results.md), [023](experiment-023-echonext-readout-results.md),
[024](experiment-024-embedding-geometry-results.md), [025](experiment-025-label-efficiency-results.md),
[026](experiment-026-normal-manifold-results.md), [027](experiment-027-calibrated-threshold-results.md) and
[028](experiment-028-echo-label-efficiency-results.md), the [clinician review sheet](experiment-024-clinician-review.md),
and the backlog entries `site_recalibration` and `local_normal_manifold` in
[experiment-backlog.json](experiment-backlog.json).
