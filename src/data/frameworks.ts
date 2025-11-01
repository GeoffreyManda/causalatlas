export interface FrameworkTopic {
  id: string;
  name: string;
  title: string;
  description: string;
  content: string;
  keyFeatures: string[];
  whenToUse: string[];
  examples: string;
}

export const frameworks: FrameworkTopic[] = [
  {
    id: 'potential-outcomes',
    name: 'PotentialOutcomes',
    title: 'Potential Outcomes Framework',
    description: 'The Rubin Causal Model: comparing what actually happened with what would have happened under alternative treatments',
    content: `The Potential Outcomes framework, also known as the **Rubin Causal Model**, is the most widely used approach for defining causal effects.

## Core Idea

For each unit *i* and treatment *a*, there exists a potential outcome **Y^a_i** - the outcome that would be observed if unit *i* received treatment *a*. The causal effect is the difference between potential outcomes under different treatments.

## Key Notation

- **Y^1**: Potential outcome under treatment (A=1)
- **Y^0**: Potential outcome under control (A=0)  
- **Y**: Observed outcome (Y = Y^1·A + Y^0·(1-A))
- **Individual Treatment Effect**: ITE_i = Y^1_i - Y^0_i
- **Average Treatment Effect**: ATE = E[Y^1 - Y^0]

## Fundamental Problem

We only observe one potential outcome per unit - the **fundamental problem of causal inference**. If unit *i* receives treatment (A_i=1), we observe Y^1_i but not Y^0_i.

## Core Assumptions

**1. SUTVA** (Stable Unit Treatment Value Assumption)
- No interference between units
- Treatment is well-defined (no hidden variations)

**2. Consistency**: Y = Y^A (the observed outcome equals the potential outcome under the actual treatment)

**3. Ignorability/Exchangeability**: (Y^1, Y^0) ⊥ A | X (treatment assignment is independent of potential outcomes given covariates)

**4. Positivity**: 0 < P(A=1|X=x) < 1 for all x (all covariate patterns have positive probability of both treatments)`,
    keyFeatures: [
      'Defines causal effects as comparisons of potential outcomes',
      'Makes the fundamental problem of causal inference explicit',
      'Focuses on what would happen under intervention',
      'Natural for randomized and observational studies',
      'Strong tradition in statistics and epidemiology'
    ],
    whenToUse: [
      'When you want to compare treatment regimes',
      'Population-level effect estimation (ATE, ATT)',
      'Heterogeneous treatment effects',
      'Missing data and selection bias problems',
      'Time-varying treatments with g-formula'
    ],
    examples: `**Example: Job Training Program**

Units: Individuals  
Treatment A: Job training (1) vs. no training (0)  
Outcome Y: Earnings 1 year later

**Potential Outcomes:**
- Y^1_i: Earnings if person *i* received training
- Y^0_i: Earnings if person *i* did not receive training

**Causal Effect:** ITE_i = Y^1_i - Y^0_i (usually unobservable)

**Average Treatment Effect:** ATE = E[Y^1 - Y^0]

If we randomly assign training, then (Y^1, Y^0) ⊥ A, so:
ATE = E[Y|A=1] - E[Y|A=0]`
  },
  {
    id: 'scm',
    name: 'SCM',
    title: 'Structural Causal Models (SCM)',
    description: 'Pearl\'s approach using directed acyclic graphs and structural equations to represent causal mechanisms',
    content: `Structural Causal Models (SCMs), developed by Judea Pearl, use **directed acyclic graphs (DAGs)** and **structural equations** to represent causal mechanisms explicitly.

## Core Components

An SCM consists of:

**1. Endogenous variables** (V): Variables whose values are determined within the model

**2. Exogenous variables** (U): Background factors (unobserved randomness, noise)

**3. Structural equations**: Each variable V_i is a function of its parents and noise:
   V_i = f_i(PA_i, U_i)

**4. DAG**: Graph where X → Y means X is a direct cause of Y

## The do-operator

The **do-operator** represents intervention: *do(X=x)* means we surgically set X to x by:
- Removing all arrows into X
- Fixing X = x

This differs from conditioning P(Y|X=x), which is passive observation.

**Key Distinction:**
- P(Y|X=x): Probability of Y given we observed X=x  
- P(Y|do(X=x)): Probability of Y if we intervene to set X=x

## Identification via d-separation

Pearl's theory provides graphical criteria (backdoor, frontdoor) to determine when causal effects are identifiable from observational data.

**Backdoor Criterion:** Control for confounders that block all backdoor paths from treatment to outcome

**Frontdoor Criterion:** Use mediators when confounders are unmeasured

## Connection to Potential Outcomes

The do-operator corresponds to potential outcomes: P(Y=y|do(A=a)) = P(Y^a=y)

SCM provides the graph, PO provides the estimands.`,
    keyFeatures: [
      'Explicit causal graphs (DAGs) show assumptions visually',
      'do-calculus provides formal rules for causal reasoning',
      'Mediation analysis naturally represented',
      'Transportability and selection bias have graphical criteria',
      'Counterfactuals defined via structural equations'
    ],
    whenToUse: [
      'When you need to reason about complex causal structures',
      'Mediation and path-specific effects',
      'Transportability and generalization',
      'When assumptions need to be transparent and visual',
      'Sensitivity analysis using unmeasured confounding'
    ],
    examples: `**Example: Smoking → Lung Cancer**

**DAG:**
Smoking → Tar deposits → Lung cancer  
Genetics → Smoking  
Genetics → Lung cancer

**Structural equations:**
Smoking = f₁(Genetics, U₁)  
Tar = f₂(Smoking, U₂)  
Cancer = f₃(Tar, Genetics, U₃)

**Intervention:** do(Smoking=1) removes the arrow Genetics → Smoking

**Causal effect:** P(Cancer|do(Smoking=1)) - P(Cancer|do(Smoking=0))

**Backdoor path:** Smoking ← Genetics → Cancer (blocked by controlling Genetics)`
  },
  {
    id: 'principal-stratification',
    name: 'PrincipalStratification',
    title: 'Principal Stratification',
    description: 'Framework for handling post-treatment variables and non-compliance by stratifying on latent principal strata',
    content: `**Principal Stratification**, introduced by Frangakis and Rubin, defines causal effects within subgroups (strata) defined by potential values of post-treatment variables.

## The Problem

Standard causal inference assumes we can control treatment directly. But often:
- **Non-compliance**: Subjects don't take assigned treatment
- **Truncation by death**: Outcome undefined for some units (e.g., quality of life if patient dies)
- **Censoring**: Outcome missing for some

Conditioning on observed post-treatment variables introduces **post-treatment bias**.

## Principal Strata

Define strata based on **joint potential values** of a post-treatment variable S under all treatments:

**Example (non-compliance):**
- **Compliers**: S^0=0, S^1=1 (take treatment if assigned, not otherwise)
- **Always-takers**: S^0=1, S^1=1 (always take treatment)
- **Never-takers**: S^0=0, S^1=0 (never take treatment)
- **Defiers**: S^0=1, S^1=0 (do opposite of assignment)

Strata are **pre-treatment** characteristics (defined before randomization), even though they involve post-treatment variables.

## Causal Effects Within Strata

**Complier Average Causal Effect (CACE):**
CACE = E[Y^1 - Y^0 | Compliers]

This is the effect of treatment *for those who would comply*.

## Identification

Under **monotonicity** (no defiers) and **exclusion restriction** (treatment assignment affects outcome only through treatment received), CACE is identified by instrumental variables:

CACE = (E[Y|Z=1] - E[Y|Z=0]) / (E[D|Z=1] - E[D|Z=0])

where Z is randomized assignment, D is treatment received.`,
    keyFeatures: [
      'Handles post-treatment confounding rigorously',
      'Defines meaningful causal effects in presence of non-compliance',
      'Extends to truncation by death and other complex scenarios',
      'Principal strata are pre-treatment (not biased)',
      'Natural for instrumental variables'
    ],
    whenToUse: [
      'Non-compliance in randomized trials',
      'Truncation by death (survivor causal effects)',
      'Mediation with treatment-confounder feedback',
      'Analyzing instrumental variables',
      'Quality of life outcomes when death is possible'
    ],
    examples: `**Example: Flu Vaccine Trial**

**Randomization (Z):** Vaccine offered (Z=1) vs. not (Z=0)  
**Treatment (D):** Vaccine received (D=1) or not (D=0)  
**Outcome (Y):** Flu infection (Y=1) or not (Y=0)

**Principal Strata:**
- Compliers: Would take vaccine if offered, not otherwise
- Always-takers: Would find vaccine regardless
- Never-takers: Would refuse even if offered

**CACE = E[Y^1 - Y^0 | Compliers]**: Effect of vaccine for those who comply with assignment

**Identification:** Under monotonicity (no one does opposite) and exclusion restriction (offer affects flu only through vaccination):

CACE = (P(Flu|Z=1) - P(Flu|Z=0)) / (P(Vaccinated|Z=1) - P(Vaccinated|Z=0))`
  },
  {
    id: 'proximal-negative-control',
    name: 'ProximalNegativeControl',
    title: 'Proximal Causal Inference & Negative Controls',
    description: 'Modern approaches for dealing with unmeasured confounding using proxies and negative control variables',
    content: `**Proximal Causal Inference** is a modern framework for identifying causal effects when confounders are unmeasured but we have **proxy variables** that contain information about them.

## The Unmeasured Confounding Problem

Traditional methods (IPW, AIPW) require that we measure all confounders (ignorability assumption). But often:
- Socioeconomic status affects both treatment and outcome, but is hard to measure fully
- Genetic factors or disease severity may be unmeasured
- Healthcare utilization reflects unmeasured health status

## Proximal Framework

Instead of measuring confounder U directly, we use:

**1. Treatment confounder proxy (Z):** A variable that contains information about U and affects treatment
   Z ← U → A

**2. Outcome confounder proxy (W):** A variable that contains information about U and affects outcome
   W ← U → Y

**3. Completeness assumptions:** Z and W together capture all the confounding information in U

Under these assumptions, causal effects are identified using two-stage methods that leverage both proxies.

## Negative Control Variables

**Negative control outcome:** An outcome that is affected by confounders but NOT by the treatment
**Negative control exposure:** An exposure that is affected by confounders but does NOT affect the outcome

These variables help:
- Detect unmeasured confounding
- Calibrate for bias
- Enable identification under additional assumptions

## Bridge Functions

Proximal methods use **bridge functions** (conditional densities) to "deconfound":

h₀(a,w) = E[Y|A=a,W=w,U] / P(A=a|Z,U)

This function bridges the gap created by unmeasured U.`,
    keyFeatures: [
      'Allows causal inference with unmeasured confounding',
      'Uses proxy variables that are easier to measure',
      'Negative controls help detect and adjust for bias',
      'Modern doubly-robust estimators available',
      'Extends to time-varying treatments'
    ],
    whenToUse: [
      'When key confounders are unmeasured',
      'Electronic health records with utilization proxies',
      'When you have instruments or negative controls',
      'Sensitivity analysis for unmeasured confounding',
      'Complex confounding structures'
    ],
    examples: `**Example: Drug Effect on Mortality (EHR Data)**

**Problem:** Disease severity (U) affects both drug prescription (A) and mortality (Y), but is unmeasured.

**Proxies:**
- **Z (treatment proxy):** Lab tests ordered (reflects severity, affects prescription)
- **W (outcome proxy):** Number of hospitalizations (reflects severity, affects mortality)

**Negative control outcome:** Dental visits (affected by health status, not by drug)

**Identification:** Use Z and W to construct bridge functions that adjust for unmeasured severity.

**Estimator:** Doubly-robust proximal causal estimator combining Z and W

This allows estimating E[Y^1 - Y^0] even though we never measured U (disease severity) directly.`
  },
  {
    id: 'bayesian-decision',
    name: 'BayesianDecision',
    title: 'Bayesian Decision-Theoretic Framework',
    description: 'Decision-theoretic approach combining causal inference with utility functions for optimal treatment decisions',
    content: `The **Bayesian Decision-Theoretic Framework** extends causal inference by incorporating **uncertainty** and **utility** to make optimal treatment decisions.

## Beyond Point Estimates

Traditional causal inference focuses on estimands like ATE. But decision-makers need:
- **Uncertainty quantification:** Not just point estimates, but full posterior distributions
- **Loss functions:** Different errors have different costs
- **Personalized decisions:** Optimal treatment may vary by individual
- **Sequential decisions:** Learning from past data and adapting

## Components

**1. Prior beliefs:** π(θ) - what we believe before seeing data

**2. Data model:** P(Data|θ) - how data arise given parameters

**3. Posterior:** P(θ|Data) ∝ P(Data|θ)π(θ) - updated beliefs

**4. Utility function:** U(action, outcome) - value of outcomes

**5. Expected utility:** EU(a) = ∫ U(a, y) P(Y=y|do(A=a), Data) dy

**Optimal decision:** Choose action *a* that maximizes expected utility

## Posterior Predictive for Counterfactuals

The posterior predictive distribution for potential outcome Y^a is:

P(Y^a | Data) = ∫ P(Y^a | θ) P(θ | Data) dθ

This provides full uncertainty about what would happen under treatment *a*.

## Value of Information

**Expected Value of Perfect Information (EVPI):** How much would we pay to remove all uncertainty?

**Expected Value of Sample Information (EVSI):** How much is an additional study worth?

These guide adaptive trial designs and research prioritization.`,
    keyFeatures: [
      'Full uncertainty quantification via posterior distributions',
      'Incorporates decision-makers utility functions',
      'Natural for personalized treatment decisions',
      'Enables adaptive trial designs and sequential decisions',
      'Principled approach to value of information'
    ],
    whenToUse: [
      'When decisions must account for uncertainty',
      'Personalized/precision medicine',
      'Adaptive clinical trial designs',
      'When different errors have different costs',
      'Sequential decision-making (reinforcement learning)'
    ],
    examples: `**Example: Precision Medicine**

**Setting:** Choosing treatment *a* ∈ {0,1} for patient with covariates *x*.

**Utility:** U(a, Y^a) = -|Y^a - target|² (minimize squared error from health target)

**Posterior:** Based on previous patient data, estimate P(Y^a | x, Data)

**Expected Utility:**
EU(a|x) = ∫ U(a, y) P(Y^a=y | x, Data) dy

**Optimal treatment:** a*(x) = argmax_a EU(a|x)

**Benefit:** Accounts for both the expected outcome AND the uncertainty. If two treatments have similar expected outcomes but one has high uncertainty, the certain one may be preferred (risk aversion).

**Adaptive trials:** Update posterior as data accumulates, adapt randomization probabilities toward more promising treatments.`
  }
];
