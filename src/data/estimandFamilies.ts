export interface EstimandFamilyTopic {
  id: string;
  name: string;
  title: string;
  description: string;
  content: string;
  keyQuestions: string[];
  commonEstimands: string[];
  example: string;
}

export const estimandFamilies: EstimandFamilyTopic[] = [
  {
    id: 'population-effects',
    name: 'PopulationEffects',
    title: 'Population-Level Treatment Effects',
    description: 'Average causal effects in populations or subgroups',
    content: `**Population Effects** quantify the **average causal effect** of a treatment across a population or specific subgroup.

## Core Estimands

**1. Average Treatment Effect (ATE)**
- **Definition:** E[Y^1 - Y^0]
- **Interpretation:** Expected difference in outcomes if everyone in the population received treatment vs. control
- **Use:** Policy-making, population health interventions

**2. Average Treatment Effect on the Treated (ATT)**
- **Definition:** E[Y^1 - Y^0 | A=1]
- **Interpretation:** Effect for those who actually received treatment
- **Use:** Evaluating existing programs, observational studies

**3. Average Treatment Effect on the Untreated (ATU)**
- **Definition:** E[Y^1 - Y^0 | A=0]
- **Interpretation:** What effect would controls have experienced if treated?
- **Use:** Expansion of programs to new populations

**4. Conditional Average Treatment Effect (CATE)**
- **Definition:** E[Y^1 - Y^0 | X=x]
- **Interpretation:** Effect for subgroup with characteristics X=x
- **Use:** Personalized medicine, targeting interventions

## Why Population Effects?

Population effects answer **"what if everyone..."** questions:
- What if we ban smoking nationwide?
- What if we treat all diabetics with this drug?
- What if we expand Medicaid to more states?

They aggregate individual effects into policy-relevant summaries.`,
    keyQuestions: [
      'What is the average effect of treatment in the population?',
      'Does the treatment work better for certain subgroups?',
      'Should we expand this intervention to new populations?',
      'What is the effect for those who are currently treated?',
      'How much heterogeneity is there in treatment effects?'
    ],
    commonEstimands: [
      'Average Treatment Effect (ATE)',
      'Average Treatment Effect on Treated (ATT)',
      'Conditional Average Treatment Effect (CATE)',
      'Quantile Treatment Effects',
      'Weighted Average Treatment Effects'
    ],
    example: `**Example: Job Training Program**

**ATE = E[Y^1 - Y^0] = $3,500**
- If we enrolled everyone (treated and controls) in the program, average earnings would increase by $3,500

**ATT = E[Y^1 - Y^0 | A=1] = $4,200**
- For those who were actually trained, the effect is $4,200 (maybe they were more motivated)

**CATE = E[Y^1 - Y^0 | Education = High School] = $2,800**
- For high school graduates, effect is $2,800

**CATE = E[Y^1 - Y^0 | Education = College] = $1,500**
- For college graduates, effect is smaller (they have better jobs already)`
  },
  {
    id: 'instrumental-local',
    name: 'InstrumentalLocal',
    title: 'Instrumental Variables & Local Effects',
    description: 'Exploiting instruments to estimate effects in the presence of unmeasured confounding',
    content: `**Instrumental Variable (IV)** methods estimate causal effects when there is **unmeasured confounding**, by leveraging an **instrument** - a variable that affects treatment but not the outcome directly.

## What is an Instrument (Z)?

An instrument must satisfy:

**1. Relevance:** Z affects treatment A
   - Cor(Z, A) ≠ 0

**2. Exclusion Restriction:** Z affects outcome Y only through treatment A
   - Z → A → Y (no direct path Z → Y)

**3. Exchangeability:** Z is unconfounded
   - (Y^z, A^z) ⊥ Z

**Examples of instruments:**
- Randomized encouragement (nudges)
- Distance to treatment facility
- Genetic variants (Mendelian randomization)
- Policy changes that affect treatment access

## Local Average Treatment Effect (LATE)

IV methods typically estimate the **LATE** (also called Complier Average Causal Effect, CACE):

**LATE = E[Y^1 - Y^0 | Compliers]**

**Compliers:** Those whose treatment status would change if instrument changed
- If Z=1 → A=1, if Z=0 → A=0

**Formula (binary Z, A):**
LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[A|Z=1] - E[A|Z=0])

**Numerator:** Intent-to-treat effect (effect of instrument on outcome)
**Denominator:** Effect of instrument on treatment (first-stage)

## Why "Local"?

LATE is the effect **for compliers only**, not the entire population. This is a more limited (local) effect than ATE.

## Assumptions

**Monotonicity:** No defiers (nobody does the opposite of what instrument suggests)

**Strong first stage:** Instrument actually affects treatment (testable)

**Exclusion restriction:** Instrument affects outcome only through treatment (not testable)`,
    keyQuestions: [
      'What is the effect of treatment when confounders are unmeasured?',
      'Can we use natural variation (instrument) to estimate causal effects?',
      'What is the effect for those whose treatment is affected by the instrument?',
      'Is the instrument valid (exclusion restriction)?',
      'Is the first stage strong enough?'
    ],
    commonEstimands: [
      'Local Average Treatment Effect (LATE)',
      'Complier Average Causal Effect (CACE)',
      'Marginal Treatment Effect (MTE)',
      'Local Average Response Function (LARF)'
    ],
    example: `**Example: Effect of Military Service on Earnings**

**Problem:** Veterans may differ from non-veterans in unmeasured ways (motivation, health) that affect earnings.

**Instrument (Z):** Vietnam War draft lottery number
- Low number → drafted (A=1)
- High number → not drafted (A=0)

**Why valid instrument?**
- **Relevant:** Lottery number strongly predicts service
- **Exchangeable:** Lottery is random
- **Exclusion:** Lottery number affects earnings only through service (not through other channels)

**Principal Strata:**
- **Compliers:** Served because drafted, wouldn't have served otherwise
- **Always-takers:** Volunteered regardless of draft
- **Never-takers:** Avoided service even if drafted (medical, educational deferments)

**LATE:** Effect of service for compliers ≈ -$2,000 per year (negative effect on earnings)

**Interpretation:** For those induced to serve by the draft, military service reduced earnings by $2,000/year on average.

**Note:** This is NOT the effect for volunteers (always-takers).`
  },
  {
    id: 'survival-time-to-event',
    name: 'SurvivalTimeToEvent',
    title: 'Survival & Time-to-Event Analysis',
    description: 'Causal effects on time until an event occurs, accounting for censoring',
    content: `**Survival Analysis** deals with outcomes that are **time-to-event**, such as:
- Time to death
- Time to disease recurrence
- Time to hospital readmission
- Duration of unemployment

## Key Challenges

**1. Censoring:** We don't always observe the event
- **Right censoring:** Study ends before event occurs
- **Left censoring:** Event occurred before observation began
- **Interval censoring:** Event occurred between observation times

**2. Time-varying confounding:** Covariates and treatment may change over time

**3. Competing risks:** Multiple types of events can occur (death from different causes)

## Estimands

**1. Hazard Ratio (HR)**
- **Definition:** λ₁(t) / λ₀(t) where λₐ(t) = P(event at t | survived to t, A=a)
- **Interpretation:** Instantaneous risk of event under treatment vs. control
- **Causal interpretation:** Under no unmeasured confounding and proportional hazards

**2. Restricted Mean Survival Time (RMST)**
- **Definition:** E[min(T^1, τ)] - E[min(T^0, τ)]
- **Interpretation:** Difference in average survival time up to time τ
- **Advantage:** Does not assume proportional hazards; directly interpretable

**3. Survival Curve Difference**
- **Definition:** S₁(t) - S₀(t) where Sₐ(t) = P(T^a > t)
- **Interpretation:** Difference in probability of surviving past time t

**4. Median Survival Time Difference**
- **Definition:** Median(T^1) - Median(T^0)
- **Interpretation:** Difference in time by which 50% have experienced event

## Methods

**Cox proportional hazards model:** Assumes hazard ratio is constant over time

**Kaplan-Meier curves:** Non-parametric survival estimation

**G-formula:** Accounts for time-varying confounding

**Marginal Structural Models (MSM):** Inverse probability weighting over time`,
    keyQuestions: [
      'What is the effect of treatment on survival time?',
      'Does treatment change the rate of events over time?',
      'How much longer do patients survive under treatment?',
      'What is the effect accounting for competing risks?',
      'How do we handle time-varying treatments and confounders?'
    ],
    commonEstimands: [
      'Hazard Ratio (HR)',
      'Restricted Mean Survival Time (RMST)',
      'Survival Curve Difference at time t',
      'Median Survival Time Difference',
      'Cause-specific Cumulative Incidence'
    ],
    example: `**Example: Cancer Treatment and Survival**

**Treatment:** Chemotherapy (A=1) vs. standard care (A=0)

**Outcome:** Time to death (T)

**Censoring:** Some patients still alive at end of study (right censored)

**Estimands:**

**1. Hazard Ratio:** HR = 0.65 (95% CI: 0.52-0.81)
- At any time t, risk of death is 35% lower under chemotherapy

**2. RMST (5-year):** Difference = 8.2 months (95% CI: 4.1-12.3)
- Patients live 8.2 months longer on average within first 5 years

**3. Median survival:** 42 months (chemo) vs. 28 months (control)
- Median survival is 14 months longer with chemotherapy

**Kaplan-Meier Curves:**
- At 3 years: S₁(3) = 0.62, S₀(3) = 0.45
- Survival probability at 3 years is 17 percentage points higher with chemotherapy`
  },
  {
    id: 'longitudinal-dynamic',
    name: 'LongitudinalDynamic',
    title: 'Longitudinal & Dynamic Treatment Effects',
    description: 'Causal effects of time-varying treatments with time-varying confounding',
    content: `**Longitudinal causal inference** deals with treatments and covariates that **change over time**. This introduces **time-varying confounding** that standard methods cannot handle.

## The Problem: Time-Varying Confounding

**Example:** HIV treatment and CD4 count
- **Time 1:** Low CD4 → start treatment
- **Time 2:** Treatment raises CD4
- **Time 3:** Higher CD4 → continue treatment

CD4 is:
1. A **confounder** (affects treatment decision at t+1)
2. An **intermediate outcome** (affected by past treatment)

**Standard adjustment fails:** Controlling for CD4 blocks the causal path from past treatment through CD4 to future outcome.

**Solution:** G-methods (g-formula, IPW, g-estimation)

## Estimands

**1. Regime-Specific Mean Outcome**
- **Definition:** E[Y^ḡ] where ḡ = (g₀, g₁, ..., gₜ) is a treatment regime
- **Example:** E[Y^{always treat}] - E[Y^{never treat}]

**2. Optimal Dynamic Treatment Regime (DTR)**
- **Definition:** d* = argmax_d E[Y^d] where d is a decision rule d(L̄ₜ)
- **Interpretation:** The best treatment strategy based on evolving patient characteristics

**3. Marginal Structural Model (MSM) Parameters**
- **Definition:** E[Y^ā] = β₀ + β₁·a + β₂·f(ā, t)
- **Interpretation:** β₁ is the effect of treatment at each time

## Methods

**G-formula (standardization):**
- Simulate outcomes under each treatment regime
- Average over covariate distribution

**Inverse Probability Weighting (IPW):**
- Weight observations by inverse probability of treatment history
- Creates pseudo-population where treatment is randomized

**G-estimation:**
- Estimates Structural Nested Models (SNM)
- Directly models treatment effect given history`,
    keyQuestions: [
      'What is the effect of sustained treatment over time?',
      'What is the optimal treatment strategy?',
      'How do we handle time-varying confounders affected by treatment?',
      'What is the effect of different treatment regimes?',
      'How do effects accumulate over time?'
    ],
    commonEstimands: [
      'Effect of always treat vs. never treat',
      'Effect of dynamic treatment regimes',
      'Marginal Structural Model (MSM) parameters',
      'Optimal DTR value function',
      'Cumulative treatment effects'
    ],
    example: `**Example: HIV Treatment Regimes**

**Setting:** Patients followed over 10 years, treatment decisions every 6 months

**Treatment (Aₜ):** Start/continue ART (1) or not (0) at time t

**Covariates (Lₜ):** CD4 count, viral load at time t (time-varying confounders)

**Outcome (Y):** AIDS-free survival at 10 years

**Regimes:**
- **g₁:** Always treat
- **g₂:** Never treat
- **g₃:** Treat if CD4 < 350

**Estimand:** E[Y^{g₁}] - E[Y^{g₂}]

**Method:** IPW for MSM
- Weight each person by inverse probability of their treatment history
- Estimate: E[Y^{always treat}] = 0.88, E[Y^{never treat}] = 0.62
- **Effect:** Always treating increases 10-year AIDS-free survival by 26 percentage points

**Optimal DTR:** Found that treating when CD4 < 500 (not 350) is optimal.`
  },
  {
    id: 'deep-representation',
    name: 'DeepRepresentation',
    title: 'Deep Learning for Representation & Causal Inference',
    description: 'Using neural networks to learn balanced representations for robust causal effect estimation',
    content: `**Deep learning** methods learn **representations** of covariates that facilitate causal inference, especially when:
- High-dimensional covariates (images, text, genomics)
- Complex confounding relationships
- Need for flexible function approximation

## Core Idea

**Standard methods** (IPW, regression) require:
1. Modeling propensity score P(A|X) and/or
2. Modeling outcome regression E[Y|A,X]

Both can be misspecified with complex X.

**Deep representation learning:**
Learn a representation Φ(X) that:
1. **Balances** treatment groups (reduces confounding)
2. **Preserves information** about outcomes
3. **Enables robust** causal effect estimation

## Methods

**1. TARNet / CFR (Counterfactual Regression)**
- Learn shared representation Φ(X)
- Separate heads for μ₀(Φ) and μ₁(Φ)
- **Objective:** Minimize prediction error + representation balance

**2. Dragonnet**
- Architecture inspired by doubly-robust estimation
- Three heads: propensity e(X), outcome μ₀(X), μ₁(X)
- **Advantage:** Mimics AIPW (doubly-robust)

**3. CEVAE (Causal Effect VAE)**
- Variational autoencoder for causal inference
- Learns latent confounder representation
- Generates counterfactuals

**4. Causal Transformers**
- Use attention mechanisms to handle sequential treatments
- Learn representations of treatment histories
- Applications in longitudinal data

## Loss Functions

**Representation balance:** Minimize distributional distance between Φ(X|A=0) and Φ(X|A=1)
- Wasserstein distance
- Maximum Mean Discrepancy (MMD)

**Outcome prediction:** Minimize factual error ∑ᵢ (Yᵢ - μ_{Aᵢ}(Φ(Xᵢ)))²

**Combined:** L = L_pred + λ·L_balance`,
    keyQuestions: [
      'How do we handle high-dimensional confounders (images, text)?',
      'Can we learn representations that are balanced between treatment groups?',
      'What is the causal effect when covariates are complex?',
      'How do we combine deep learning with causal inference theory?',
      'Can we generate counterfactual predictions using neural networks?'
    ],
    commonEstimands: [
      'Average Treatment Effect (ATE)',
      'Conditional Average Treatment Effect (CATE)',
      'Individual Treatment Effect (ITE)',
      'Counterfactual predictions Ŷ^a(x)'
    ],
    example: `**Example: Causal Effect of ICU Stay on Mortality (EHR Data)**

**Problem:** High-dimensional confounders
- Lab results, vitals, medications, demographics
- Complex interactions and non-linearities

**Standard methods:** Difficult to specify propensity and outcome models correctly

**TARNet approach:**
1. **Input:** X = [demographics, labs, vitals, medications]
2. **Representation:** Φ(X) = 64-dim vector (learned by neural network)
3. **Treatment balance:** Minimize MMD(Φ(X|A=1), Φ(X|A=0))
4. **Outcome heads:** μ₀(Φ) and μ₁(Φ) (separate neural networks)
5. **Prediction:** ̂τ(x) = μ₁(Φ(x)) - μ₀(Φ(x))

**Result:** TARNet ATE estimate = -0.08 (95% CI: -0.12, -0.04)
- ICU stay reduces mortality by 8 percentage points

**Advantage:** Flexible modeling + representation balance → more robust than misspecified logistic/linear models`
  }
];
