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
  }
];
