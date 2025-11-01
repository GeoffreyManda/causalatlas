export interface StudyDesignTopic {
  id: string;
  name: string;
  title: string;
  description: string;
  content: string;
  strengths: string[];
  limitations: string[];
  typicalEstimands: string[];
  example: string;
}

export const studyDesigns: StudyDesignTopic[] = [
  {
    id: 'rct-parallel',
    name: 'RCT_Parallel',
    title: 'Randomized Controlled Trial (Parallel Groups)',
    description: 'The gold standard: randomized assignment to treatment or control groups',
    content: `**Randomized Controlled Trials (RCTs)** are the gold standard for causal inference because randomization ensures treatment groups are comparable on all characteristics, measured and unmeasured.

## Design

- Units are **randomly assigned** to treatment (A=1) or control (A=0)
- Parallel groups: Each unit receives one treatment throughout
- Outcomes measured after treatment period

## Why Randomization Works

**Randomization ensures exchangeability:** (Y^1, Y^0) ⊥ A

This means:
- Treatment and control groups are balanced on all covariates (in expectation)
- Any difference in outcomes is due to treatment, not confounding
- Simple difference in means estimates the ATE: E[Y|A=1] - E[Y|A=0]

## Types

**1. Superiority trial:** Test if new treatment is better than control

**2. Non-inferiority trial:** Test if new treatment is not worse than standard (by a margin)

**3. Equivalence trial:** Test if new treatment has similar effect to standard

## Analysis

**Intention-to-Treat (ITT):** Analyze all participants as randomized, regardless of compliance
- Preserves randomization
- Estimates effect of *assignment* to treatment

**Per-Protocol:** Analyze only those who completed treatment as assigned
- Biased (conditioning on post-randomization variable)
- May overestimate efficacy`,
    strengths: [
      'Eliminates confounding through randomization',
      'Provides strongest causal evidence',
      'Simple analysis (difference in means)',
      'No unmeasured confounding',
      'Enables blind/double-blind designs'
    ],
    limitations: [
      'Expensive and time-consuming',
      'May not be ethical or feasible for some treatments',
      'External validity: RCT participants may differ from target population',
      'Non-compliance can complicate interpretation',
      'Limited to estimating average effects in the trial population'
    ],
    typicalEstimands: [
      'Average Treatment Effect (ATE)',
      'Complier Average Causal Effect (CACE)',
      'Quantile Treatment Effects',
      'Subgroup-specific effects'
    ],
    example: `**Example: New Diabetes Drug Trial**

**Design:** 400 patients randomized 1:1 to drug (A=1) or placebo (A=0)

**Outcome:** HbA1c reduction at 12 weeks

**Randomization:** Ensures treated and control groups are balanced on:
- Age, sex, baseline HbA1c, disease duration, comorbidities
- Unmeasured factors (genetics, medication adherence, lifestyle)

**Analysis:** 
ITT estimate: Mean(Y|A=1) - Mean(Y|A=0) = -0.8% (95% CI: -1.1%, -0.5%)

**Interpretation:** Drug reduces HbA1c by 0.8 percentage points on average, compared to placebo.`
  },
  {
    id: 'cohort',
    name: 'Cohort',
    title: 'Cohort Study (Observational)',
    description: 'Following groups exposed and unexposed to a treatment over time',
    content: `**Cohort studies** follow groups of individuals over time to compare outcomes between those exposed and unexposed to a treatment.

## Design

**1. Define cohort:** A group of individuals followed over time

**2. Measure exposure:** Treatment/exposure status at baseline (or time-varying)

**3. Follow-up:** Observe outcomes over time

**4. Compare:** Outcome incidence/levels in exposed vs. unexposed

## Types

**Prospective cohort:** Exposure measured now, follow forward in time
- More expensive, but better data quality
- Can measure exposures and covariates precisely

**Retrospective cohort:** Use existing records, look back at exposure and outcomes
- Cheaper and faster
- Limited by available data

## Confounding

**Key challenge:** Exposure is not randomized, so exposed and unexposed may differ in many ways.

**Solution:** Adjust for measured confounders using:
- Regression models
- Propensity score methods (IPW, matching, stratification)
- Doubly-robust estimators (AIPW, TMLE)

**Assumptions:** To identify causal effects, must assume:
- **Exchangeability given X:** (Y^1, Y^0) ⊥ A | X (no unmeasured confounding)
- **Positivity:** P(A=1|X=x) > 0 for all x
- **Consistency and SUTVA**

## Advantages over Case-Control

- Can estimate incidence (risk) directly
- Can study multiple outcomes
- Temporal sequence clear (exposure before outcome)`,
    strengths: [
      'Feasible for exposures that cannot be randomized',
      'Can study rare exposures',
      'Can examine multiple outcomes',
      'Temporal sequence is clear',
      'Less expensive than RCTs'
    ],
    limitations: [
      'Confounding: Exposed and unexposed groups may differ systematically',
      'Requires measuring all confounders (strong ignorability assumption)',
      'Loss to follow-up can introduce bias',
      'Expensive and time-consuming for rare outcomes',
      'Temporal ambiguity if exposure is not fixed at baseline'
    ],
    typicalEstimands: [
      'Average Treatment Effect (ATE)',
      'Average Treatment Effect on Treated (ATT)',
      'Conditional ATE (CATE)',
      'Risk ratios, Odds ratios (after transformation)'
    ],
    example: `**Example: Hormone Therapy and Cardiovascular Disease**

**Cohort:** 50,000 postmenopausal women in Nurses' Health Study

**Exposure:** Hormone replacement therapy (HRT) use

**Outcome:** Coronary heart disease incidence over 10 years

**Confounders:** Age, smoking, BMI, hypertension, family history, physical activity

**Analysis:** Adjust for confounders using Cox proportional hazards model or IPW

**Result:** After adjustment, HR = 1.2 (95% CI: 1.0-1.4) for HRT vs. no HRT

**Note:** Observational evidence suggested harm, later confirmed by RCT (WHI trial).`
  },
  {
    id: 'cluster-rct',
    name: 'Cluster_RCT',
    title: 'Cluster Randomized Trial',
    description: 'Randomizing groups (clusters) rather than individuals',
    content: `**Cluster Randomized Trials** randomize groups (clusters) of individuals rather than individuals themselves.

## Design

**Clusters** are randomized to treatment or control:
- Schools, hospitals, clinics, villages, communities
- All individuals within a cluster receive the same treatment

**Why use cluster randomization?**
- Intervention is naturally delivered at group level (e.g., policy change)
- Individual randomization would lead to contamination/interference
- Logistical or ethical reasons

## Statistical Issues

**1. Intra-cluster correlation (ICC):** Individuals within a cluster are more similar to each other than to individuals in other clusters.

**Effect:** Reduces effective sample size
- If ICC = 0 (no clustering), effective N = total N
- If ICC > 0, effective N < total N

**Design effect:** 1 + (m-1)·ICC, where m = cluster size

**2. Need to account for clustering in analysis:**
- Generalized Estimating Equations (GEE)
- Mixed effects models (random intercepts for clusters)
- Cluster-robust standard errors

## Sample Size

Must randomize enough **clusters** (not just enough individuals).

**Formula:** For power 80%, significance 0.05:
Number of clusters per arm ≈ (Zα/2 + Zβ)² · (1 + (m-1)·ICC) · σ² / (m·δ²)

where δ = effect size, m = avg cluster size`,
    strengths: [
      'Feasible when intervention is at group level',
      'Avoids contamination between treatment arms',
      'May be more acceptable ethically or logistically',
      'Can estimate community-level effects',
      'Pragmatic: Reflects real-world implementation'
    ],
    limitations: [
      'Requires large number of clusters for adequate power',
      'Correlation within clusters reduces effective sample size',
      'Imbalance in cluster sizes complicates analysis',
      'More complex sample size calculations',
      'May have limited generalizability if clusters are unique'
    ],
    typicalEstimands: [
      'Average Treatment Effect (cluster-level)',
      'Marginal ATE (individual-level)',
      'Within-cluster effects',
      'Between-cluster variance'
    ],
    example: `**Example: School-Based Nutrition Intervention**

**Design:** 40 schools randomized to nutrition education program (20 schools) vs. standard curriculum (20 schools)

**Outcome:** BMI change in students after 1 year

**Clusters:** Schools (avg 200 students per school)

**ICC:** Estimated ICC = 0.05 (students in same school are more similar)

**Analysis:** Mixed effects model with random intercepts for schools
- Fixed effect: Treatment (program vs. control)
- Random effect: School

**Result:** Program reduces BMI by 0.5 kg/m² (95% CI: 0.2-0.8), p=0.003

**Interpretation:** The nutrition program has a beneficial effect on student BMI, accounting for clustering within schools.`
  },
  {
    id: 'regression-discontinuity',
    name: 'Regression_Discontinuity',
    title: 'Regression Discontinuity Design',
    description: 'Exploiting treatment assignment based on a threshold of a running variable',
    content: `**Regression Discontinuity (RD)** designs exploit situations where treatment is assigned based on whether a **running variable** crosses a **threshold**.

## Design

**Running variable (X):** A continuous variable (e.g., test score, age, income)

**Cutoff (c):** A threshold value

**Treatment assignment rule:**
- **Sharp RD:** A = 1{X ≥ c} (deterministic)
- **Fuzzy RD:** P(A=1|X) jumps at X=c (probabilistic)

**Causal effect:** Local Average Treatment Effect (LATE) at the cutoff

## Intuition

**Near the cutoff**, individuals just above and just below are nearly identical (random) except for treatment assignment.

**Example:** Scholarship awarded if test score ≥ 70
- Student with score 69.5 vs. 70.5 are very similar
- Any jump in outcome at the cutoff is due to treatment (scholarship)

## Assumptions

**1. Continuity of potential outcomes:** E[Y^0|X] and E[Y^1|X] are continuous at X=c
   - No jumps in outcomes except through treatment

**2. No manipulation:** Units cannot precisely control X to just cross cutoff
   - Test density of X at cutoff (McCrary test)

**3. No other treatments at cutoff:** Treatment is the only thing that changes at c

## Estimation

Fit regression on either side of cutoff:
E[Y|X] = α + β·A + f(X-c) + γ·A·f(X-c)

**LATE at cutoff:** β = E[Y^1 - Y^0 | X=c]

**Bandwidth choice:** How close to cutoff to include data (bias-variance tradeoff)`,
    strengths: [
      'Transparent: Cutoff rule is clear',
      'No unmeasured confounding near cutoff (local randomization)',
      'Does not require all confounders to be measured',
      'Visual: Can plot outcome vs. running variable',
      'Credible causal inference from observational data'
    ],
    limitations: [
      'Only estimates effect at the cutoff (local effect, not average)',
      'Limited generalizability beyond cutoff region',
      'Requires large sample size near cutoff',
      'Sensitive to bandwidth and functional form',
      'Manipulation of running variable would invalidate design'
    ],
    typicalEstimands: [
      'Local Average Treatment Effect (LATE) at cutoff',
      'Fuzzy RD: LATE for compliers at cutoff',
      'Heterogeneity: LATE at different cutoffs'
    ],
    example: `**Example: Financial Aid and College Completion**

**Running variable:** Family income (centered at cutoff)

**Cutoff:** c = $50,000 (families below get aid)

**Treatment:** Financial aid (A=1 if income < c, A=0 otherwise)

**Outcome:** College degree completion (Y=1 if completed)

**RD estimate:**
- Plot: Degree completion vs. income
- Jump at $50,000 cutoff ≈ 10 percentage points
- Students just below cutoff are 10% more likely to complete degree

**Interpretation:** Financial aid increases completion probability by 10% for families near the income cutoff.

**Limitation:** Effect may differ for families far above or below cutoff (not generalizable).`
  },
  {
    id: 'stepped-wedge',
    name: 'Stepped_Wedge',
    title: 'Stepped Wedge Design',
    description: 'Sequential rollout where all clusters eventually receive treatment',
    content: `**Stepped Wedge** designs are a type of cluster randomized trial where clusters transition from control to treatment in **waves** over time. Eventually, all clusters receive the treatment.

## Design

**Time periods:** Divide study into T time periods

**Clusters:** K clusters (e.g., hospitals, clinics)

**Rollout:** At each time point, a random subset of control clusters transitions to treatment

**End result:** All clusters treated by end of study

## Example Timeline (K=6 clusters, T=4 periods)

| Cluster | Period 1 | Period 2 | Period 3 | Period 4 |
|---------|----------|----------|----------|----------|
| 1       | Control  | Control  | Treated  | Treated  |
| 2       | Control  | Treated  | Treated  | Treated  |
| 3       | Control  | Control  | Control  | Treated  |
| 4       | Control  | Treated  | Treated  | Treated  |
| 5       | Control  | Control  | Treated  | Treated  |
| 6       | Control  | Control  | Control  | Treated  |

Each row is a cluster; transitions are randomized.

## Advantages

**1. Ethical:** All clusters eventually benefit (no cluster permanently denied treatment)

**2. Practical:** Easier to implement intervention in phases (resource constraints)

**3. Before-after within cluster:** Can control for fixed cluster characteristics

**4. Efficiency:** Combines between-cluster and within-cluster comparisons

## Analysis

**Mixed effects model:**
Y_ijk = β₀ + β₁·Time_j + β₂·Treatment_jk + u_k + ε_ijk

- Y_ijk: Outcome for individual i in cluster k at time j
- β₂: Treatment effect (primary parameter)
- u_k: Cluster random effect
- Time_j: Time trend (important to adjust for)`,
    strengths: [
      'Ethically appealing: All clusters eventually treated',
      'Practical: Staged rollout is often feasible',
      'Efficient: Uses within-cluster comparisons',
      'Controls for time-invariant cluster characteristics',
      'Can estimate time-varying treatment effects'
    ],
    limitations: [
      'Assumes no time trend interacts with treatment',
      'Requires longer study duration',
      'More complex analysis than parallel RCT',
      'Potential for secular trends to confound results',
      'Contamination if treatment effects spill over'
    ],
    typicalEstimands: [
      'Marginal treatment effect (averaged over time and clusters)',
      'Time-varying treatment effects',
      'Cluster-specific effects'
    ],
    example: `**Example: Hand Hygiene Intervention in Hospitals**

**Design:** 12 hospitals, 6 time periods (months)

**Intervention:** Hand hygiene training program

**Outcome:** Hospital-acquired infection rate (%)

**Rollout:** Every month, 2 randomly selected hospitals that are still controls receive training.

**Analysis:** Mixed model adjusting for time trend and clustering

**Result:** Infection rate reduced by 1.5 percentage points (95% CI: 0.8-2.2) after intervention

**Interpretation:** Hand hygiene training reduces infection rate. All hospitals eventually benefit from intervention.`
  },
  {
    id: 'encouragement',
    name: 'Encouragement',
    title: 'Encouragement Design',
    description: 'Randomizing encouragement rather than treatment itself',
    content: `**Encouragement designs** randomize an **encouragement** to take treatment, rather than treatment itself. This creates an instrumental variable (IV) that can be used to estimate causal effects.

## Design

**Randomization:** Participants randomized to receive encouragement (Z=1) or not (Z=0)

**Treatment choice:** Participants decide whether to take treatment (A=1 or 0)

**Outcome:** Y measured for all participants

## Why Use Encouragement?

When direct randomization is not feasible or ethical:
- Cannot force people to exercise, quit smoking, etc.
- Cannot deny people access to available treatments
- Want to study real-world adherence

## Causal Effects

**Intent-to-Treat (ITT):** Effect of encouragement on outcome
- E[Y|Z=1] - E[Y|Z=0]

**Complier Average Causal Effect (CACE):** Effect of treatment for compliers
- CACE = ITT / (E[A|Z=1] - E[A|Z=0])
- Requires monotonicity and exclusion restriction`,
    strengths: [
      'Ethical when cannot randomize treatment directly',
      'Maintains randomization benefits',
      'Useful for interventions that require voluntary participation',
      'Can estimate treatment effects for compliers'
    ],
    limitations: [
      'Weaker instrument means larger standard errors',
      'Only identifies effect for compliers, not everyone',
      'Requires exclusion restriction (encouragement affects Y only through A)',
      'Requires monotonicity (no defiers)'
    ],
    typicalEstimands: [
      'Intent-to-Treat Effect (ITT)',
      'Complier Average Causal Effect (CACE)',
      'Local Average Treatment Effect (LATE)'
    ],
    example: `**Example: Flu Vaccination Campaign**

**Design:** 1000 employees randomized to:
- Z=1: Receive email/text reminders to get flu shot (encouragement)
- Z=0: No reminders (control)

**Treatment:** A = got flu shot (self-selected)

**Outcome:** Y = missed work days due to illness

**Results:**
- E[A|Z=1] = 0.65 (65% vaccinated in encouraged group)
- E[A|Z=0] = 0.40 (40% vaccinated in control)
- ITT = -1.2 days (encouragement reduces sick days by 1.2)
- CACE = -1.2 / (0.65-0.40) = -4.8 days

**Interpretation:** For compliers (those who got vaccinated because of encouragement), vaccination reduces sick days by 4.8 days.`
  },
  {
    id: 'case-control',
    name: 'Case_Control',
    title: 'Case-Control Study',
    description: 'Retrospective design comparing cases with disease to controls without',
    content: `**Case-Control studies** select participants based on **outcome status** (disease yes/no), then look back at exposure history. Efficient for rare diseases.

## Design

**1. Select cases:** Individuals with disease/outcome
**2. Select controls:** Individuals without disease (matched or unmatched)
**3. Measure past exposures:** Look back at exposure history
**4. Compare:** Odds of exposure in cases vs. controls

## Sampling

**Population-based:** Sample cases and controls from defined population
**Hospital-based:** Cases from hospital; controls from same hospital
**Nested case-control:** Within an existing cohort study

## Analysis

**Odds Ratio (OR):** Primary measure of association
- OR = [cases exposed / cases unexposed] / [controls exposed / controls unexposed]

**Interpretation:** Under rare disease assumption, OR approximates Risk Ratio

## Matching

Often match controls to cases on confounders (age, sex, etc.)
- Reduces confounding
- Requires conditional logistic regression for analysis`,
    strengths: [
      'Efficient for rare diseases',
      'Can study multiple exposures for one disease',
      'Faster and cheaper than cohort studies',
      'Can study diseases with long latency periods',
      'Requires fewer participants than cohort'
    ],
    limitations: [
      'Cannot estimate incidence or prevalence',
      'Prone to recall bias (differential memory of exposures)',
      'Selection of appropriate controls is challenging',
      'Can only study one outcome',
      'Temporal sequence sometimes unclear'
    ],
    typicalEstimands: [
      'Odds Ratio (OR)',
      'Attributable fraction among exposed',
      'Population Attributable Fraction (PAF)'
    ],
    example: `**Example: Lung Cancer and Smoking**

**Cases:** 500 lung cancer patients
**Controls:** 500 matched individuals without lung cancer (same age, sex, location)

**Exposure:** History of smoking

**Results:**
- Cases: 400/500 (80%) smoked
- Controls: 200/500 (40%) smoked

**Odds Ratio:** OR = (400/100) / (200/300) = 4.0 / 0.67 = 6.0

**Interpretation:** Odds of lung cancer are 6 times higher in smokers than non-smokers. Under rare disease assumption, smokers have ~6× the risk of lung cancer.`
  },
  {
    id: 'cross-sectional',
    name: 'Cross_Sectional',
    title: 'Cross-Sectional Study',
    description: 'Single time point measurement of exposure and outcome',
    content: `**Cross-Sectional studies** measure exposure and outcome at a **single time point** or over a short period. Provides a "snapshot" of the population.

## Design

**1. Define population:** Identify target population
**2. Sample participants:** Random or convenience sampling
**3. Measure simultaneously:** Exposure, outcome, covariates all at once
**4. Analyze associations:** Compare outcome prevalence across exposure groups

## Causal Interpretation

**Challenge:** Temporal ambiguity - which came first, exposure or outcome?

**When causal interpretation is more plausible:**
- Exposure is fixed (genetics, birthplace, early-life factors)
- Outcome is clearly downstream (medical diagnosis after exposure)

**When problematic:**
- Exposure and outcome could influence each other
- Reverse causation is plausible

## Analysis

**Prevalence Ratio (PR):** Prevalence of outcome in exposed / unexposed
**Prevalence Odds Ratio (POR):** Odds of outcome in exposed / unexposed

**Regression:** Logistic or log-binomial regression adjusting for confounders`,
    strengths: [
      'Quick and inexpensive to conduct',
      'Can study multiple exposures and outcomes',
      'No loss to follow-up',
      'Useful for disease prevalence estimation',
      'Hypothesis-generating for future studies'
    ],
    limitations: [
      'Cannot establish temporal sequence clearly',
      'Cannot estimate incidence',
      'Prone to reverse causation',
      'Survivor bias (only measure prevalent cases, not incident)',
      'Weaker causal inference than cohort or RCT'
    ],
    typicalEstimands: [
      'Prevalence Ratio (PR)',
      'Prevalence Odds Ratio (POR)',
      'Prevalence Difference'
    ],
    example: `**Example: Hypertension and Diabetes**

**Design:** Survey 5000 adults, measure:
- Exposure: Hypertension (yes/no)
- Outcome: Diabetes (yes/no)
- Confounders: Age, BMI, physical activity

**Results:**
- Diabetes prevalence in hypertensive: 150/1000 = 15%
- Diabetes prevalence in normotensive: 200/4000 = 5%

**PR = 15% / 5% = 3.0**

**Interpretation:** Diabetes is 3× more prevalent in hypertensive individuals. But temporal sequence is unclear - did hypertension lead to diabetes, or vice versa?`
  },
  {
    id: 'target-trial-emulation',
    name: 'Target_Trial_Emulation',
    title: 'Target Trial Emulation',
    description: 'Using observational data to emulate a hypothetical RCT',
    content: `**Target Trial Emulation** is a framework for analyzing observational data **as if** it came from a randomized trial. It forces explicit specification of trial design elements.

## Core Idea

Instead of "analyzing available data," first specify:
**1. What RCT would we ideally run?**
**2. How can we emulate it using observational data?**

This reduces common biases (immortal time, prevalent user, selection bias).

## Protocol Components

Specify explicitly:
- **Eligibility:** Who can be enrolled?
- **Treatment strategies:** What treatments to compare?
- **Assignment:** How would treatment be assigned? (In observational data: use propensity scores, IP weighting)
- **Follow-up:** When does it start? When does it end?
- **Outcome:** What is measured? When?
- **Causal contrast:** What estimand (ATE, ATT, etc.)?
- **Analysis plan:** How to adjust for confounding?

## Common Pitfalls Avoided

**Immortal time bias:** In observational data, treated group may have guaranteed survival period. TTEmulation ensures time-zero alignment.

**Prevalent user bias:** Including existing users biases effect. TTEmulation mimics new-user design.`,
    strengths: [
      'Forces transparent design specification',
      'Reduces common observational study biases',
      'Clarifies target estimand',
      'Facilitates comparison to actual RCT results',
      'Enables causal inference from EHR/claims data'
    ],
    limitations: [
      'Cannot eliminate unmeasured confounding',
      'Requires large datasets with detailed information',
      'Complexity in specifying time-zero and eligibility',
      'May not be feasible if data structure differs from trial',
      'Assumptions still required (exchangeability, positivity)'
    ],
    typicalEstimands: [
      'Average Treatment Effect (ATE)',
      'Average Treatment Effect on Treated (ATT)',
      'Survival curves under treatment strategies',
      'Cumulative incidence under strategies'
    ],
    example: `**Example: Statins and Cardiovascular Disease (Using EHR Data)**

**Target Trial Protocol:**
- **Eligibility:** Adults age 50-75 with LDL 130-190, no CVD
- **Treatment:** Statin initiation vs. no statin
- **Assignment:** Random (emulated via IP weighting)
- **Follow-up:** Starts at eligibility, 10 years
- **Outcome:** CVD event (MI, stroke, CVD death)
- **Analysis:** IP-weighted Kaplan-Meier, Cox model

**Emulation in EHR:**
- Identify eligible patients at time of first LDL 130-190
- Clone patients into two "trial arms" (statin vs. no statin)
- Censor if treatment deviates from assigned strategy
- IP weight for time-varying confounders

**Result:** HR = 0.72 (95% CI: 0.65-0.80) for statin vs. no statin

**Interpretation:** Initiating statins reduces 10-year CVD risk by 28%.`
  },
  {
    id: 'transport-frame',
    name: 'Transport_Frame',
    title: 'Transportability & Generalization',
    description: 'Extending causal estimates from trial to target population',
    content: `**Transportability** concerns extending causal effect estimates from a **source population** (e.g., RCT participants) to a **target population** (e.g., real-world patients).

## The Problem

**Internal validity:** Effect is valid in the study sample
**External validity:** Effect applies to broader population

RCTs may have limited external validity if participants differ from target population.

## Framework

**Source:** Study population (e.g., RCT participants)
**Target:** Population of interest

**Goal:** Estimate E[Y^a]_{target} using data from source

## Assumptions

**1. Conditional transportability:** Effect modification by measured covariates X
   - E[Y^a | X=x]_{source} = E[Y^a | X=x]_{target}
   
**2. Covariate distribution measured in target:** P(X)_{target}

**3. Positivity in source:** P(A=1|X) > 0 in source

## Methods

**Inverse Odds Weighting (IOW):**
- Weight source data by odds of being in target
- Creates weighted source that resembles target

**Stratification:**
- Estimate effect in source within strata of X
- Average over target distribution of X

**G-formula:**
- Estimate E[Y|A,X] in source
- Standardize to target covariate distribution`,
    strengths: [
      'Enables generalization of trial results',
      'Makes external validity assumptions explicit',
      'Can combine RCT with observational data',
      'Useful for regulatory and policy decisions',
      'Provides framework for multi-site studies'
    ],
    limitations: [
      'Requires measuring effect modifiers',
      'Assumes no unmeasured effect modification',
      'Need data on covariate distribution in target',
      'Strong assumptions hard to verify',
      'May extrapolate beyond support of data'
    ],
    typicalEstimands: [
      'Target Average Treatment Effect (TATE)',
      'Target Average Treatment Effect on Treated (TATT)',
      'Conditional effects in target subgroups'
    ],
    example: `**Example: Generalizing Diabetes Drug RCT**

**Source:** RCT participants (n=500), younger, healthier
**Target:** All diabetes patients in health system (n=50,000)

**Covariates (X):** Age, HbA1c, comorbidities, BMI

**RCT result (source):** ATE = -0.8% HbA1c reduction

**Problem:** RCT participants younger than target population, and effect may vary by age.

**Solution:** 
1. Estimate E[Y|A,X] in RCT data (source)
2. Get P(X) from EHR (target)
3. Standardize: TATE = Σₓ E[Y^1 - Y^0 | X=x]_{source} × P(X=x)_{target}

**Result:** TATE = -0.6% (smaller than RCT ATE = -0.8%)

**Interpretation:** In the full health system population, drug reduces HbA1c by 0.6%, less than the -0.8% observed in the younger, healthier RCT population.`
  },
  {
    id: 'survey-multistage',
    name: 'Survey_Multistage',
    title: 'Complex Survey Designs',
    description: 'Multi-stage sampling with clustering and stratification',
    content: `**Complex surveys** use multi-stage sampling, clustering, and stratification to efficiently sample from large populations. This creates dependencies that must be accounted for in analysis.

## Design Elements

**1. Stratification:** Divide population into strata, sample from each
   - Ensures representation of subgroups
   - Can improve precision

**2. Clustering:** Sample clusters (e.g., schools, neighborhoods), then individuals within clusters
   - Reduces costs
   - Creates intra-cluster correlation

**3. Multi-stage sampling:** 
   - Stage 1: Sample primary sampling units (PSUs)
   - Stage 2: Sample secondary units within PSUs
   - Stage 3+: Further nested sampling

**4. Sampling weights:** Probability of selection varies across individuals
   - Inverse probability of selection weights
   - Non-response adjustments
   - Post-stratification

## Analysis

**Must account for:**
- **Clustering:** Use cluster-robust standard errors, design-based variance
- **Stratification:** Include strata in analysis
- **Weights:** Use survey weights for population estimates

**Software:** Survey packages (R: survey, Stata: svy, SAS: PROC SURVEYFREQ)`,
    strengths: [
      'Efficient sampling of large, dispersed populations',
      'Ensures representation of important subgroups',
      'Cost-effective (clustering reduces travel, fieldwork)',
      'Enables population-level inference',
      'Widely used in national health surveys'
    ],
    limitations: [
      'Complex analysis required (cannot use standard methods)',
      'Clustering reduces effective sample size',
      'Requires survey weights and design information',
      'Causal inference challenging (confounding, selection)',
      'Missing data and non-response are common'
    ],
    typicalEstimands: [
      'Population prevalence',
      'Population means and totals',
      'Subgroup-specific estimates',
      'Design-based causal effects (with strong assumptions)'
    ],
    example: `**Example: National Health Survey - Smoking and Hypertension**

**Design:**
- **Stage 1:** Sample 200 counties (PSUs) with probability proportional to population
- **Stage 2:** Sample 10 census tracts per county
- **Stage 3:** Sample 20 households per tract

**Total sample:** 40,000 individuals

**Analysis:** Estimate association between smoking and hypertension

**Naive analysis (ignoring design):** OR = 1.8, SE = 0.05

**Design-based analysis (accounting for clustering, weights):** OR = 1.8, SE = 0.12

**Key difference:** Design-adjusted SE is larger (clustering reduces precision)

**Population estimate:** Among US adults, smokers have 1.8× odds of hypertension (95% CI: 1.43-2.27).`
  }
];
