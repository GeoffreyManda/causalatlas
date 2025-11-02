/**
 * Comprehensive Causal Inference Lessons
 * Based on content from:
 * - Hernán MA, Robins JM. "Causal Inference: What If." Boca Raton: Chapman & Hall/CRC, 2020.
 * - Pearl J. "Causality: Models, Reasoning, and Inference." Cambridge University Press, 2009.
 */

import { Lesson } from './lessons';

export const comprehensiveLessons: Lesson[] = [
  // FOUNDATIONAL LESSONS FROM "WHAT IF" CHAPTERS 1-3
  {
    id: 'causal-effect-definition',
    title: 'Definition of Causal Effects (What If Ch.1)',
    description: 'Individual and average causal effects using potential outcomes framework',
    category: 'theory',
    tier: 'Foundational',
    pythonCode: `# Causal Effect Definition - Potential Outcomes Framework
# Based on Hernán & Robins "What If" Chapter 1

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Simulate a population with potential outcomes
n = 1000

# Generate potential outcomes for each individual
# Y^1 = outcome if treated, Y^0 = outcome if not treated
Y1 = np.random.binomial(1, 0.6, n)  # 60% would survive if treated
Y0 = np.random.binomial(1, 0.4, n)  # 40% would survive if untreated

# Individual causal effect: Y^1 - Y^0
individual_effect = Y1 - Y0

# Average causal effect (ACE): E[Y^1 - Y^0]
ace = np.mean(Y1) - np.mean(Y0)

print("=== CAUSAL EFFECTS (POTENTIAL OUTCOMES) ===")
print(f"Sample size: {n} individuals")
print(f"\\nAverage potential outcome under treatment: E[Y^1] = {np.mean(Y1):.3f}")
print(f"Average potential outcome under no treatment: E[Y^0] = {np.mean(Y0):.3f}")
print(f"\\nAverage Causal Effect (ACE): {ace:.3f}")
print(f"Interpretation: Treatment increases survival by {ace*100:.1f} percentage points")

# Distribution of individual effects
print(f"\\nIndividual Causal Effects:")
print(f"  - Helped by treatment (Y^1=1, Y^0=0): {np.sum((Y1==1) & (Y0==0))} ({np.mean((Y1==1) & (Y0==0))*100:.1f}%)")
print(f"  - Hurt by treatment (Y^1=0, Y^0=1): {np.sum((Y1==0) & (Y0==1))} ({np.mean((Y1==0) & (Y0==1))*100:.1f}%)")
print(f"  - No effect, would survive either way: {np.sum((Y1==1) & (Y0==1))} ({np.mean((Y1==1) & (Y0==1))*100:.1f}%)")
print(f"  - No effect, would die either way: {np.sum((Y1==0) & (Y0==0))} ({np.mean((Y1==0) & (Y0==0))*100:.1f}%)")

# Key insight: We can never observe both Y^1 and Y^0 for same person!
print("\\n*** FUNDAMENTAL PROBLEM OF CAUSAL INFERENCE ***")
print("We can never observe both Y^1 and Y^0 for the same individual.")
print("This is why randomization and statistical methods are needed!")
`,
    rCode: `# Causal Effect Definition - Potential Outcomes Framework
# Based on Hernán & Robins "What If" Chapter 1

set.seed(42)

# Simulate a population with potential outcomes
n <- 1000

# Generate potential outcomes for each individual
Y1 <- rbinom(n, 1, 0.6)  # 60% would survive if treated
Y0 <- rbinom(n, 1, 0.4)  # 40% would survive if untreated

# Individual causal effect
individual_effect <- Y1 - Y0

# Average causal effect (ACE)
ace <- mean(Y1) - mean(Y0)

cat("=== CAUSAL EFFECTS (POTENTIAL OUTCOMES) ===\\n")
cat("Sample size:", n, "individuals\\n")
cat("\\nAverage potential outcome under treatment: E[Y^1] =", round(mean(Y1), 3), "\\n")
cat("Average potential outcome under no treatment: E[Y^0] =", round(mean(Y0), 3), "\\n")
cat("\\nAverage Causal Effect (ACE):", round(ace, 3), "\\n")
cat("Interpretation: Treatment increases survival by", round(ace*100, 1), "percentage points\\n")

# Distribution of individual effects
cat("\\nIndividual Causal Effects:\\n")
cat("  - Helped by treatment (Y^1=1, Y^0=0):", sum(Y1==1 & Y0==0),
    paste0("(", round(mean(Y1==1 & Y0==0)*100, 1), "%)\\n"))
cat("  - Hurt by treatment (Y^1=0, Y^0=1):", sum(Y1==0 & Y0==1),
    paste0("(", round(mean(Y1==0 & Y0==1)*100, 1), "%)\\n"))
cat("  - No effect, would survive either way:", sum(Y1==1 & Y0==1),
    paste0("(", round(mean(Y1==1 & Y0==1)*100, 1), "%)\\n"))
cat("  - No effect, would die either way:", sum(Y1==0 & Y0==0),
    paste0("(", round(mean(Y1==0 & Y0==0)*100, 1), "%)\\n"))

cat("\\n*** FUNDAMENTAL PROBLEM OF CAUSAL INFERENCE ***\\n")
cat("We can never observe both Y^1 and Y^0 for the same individual.\\n")
cat("This is why randomization and statistical methods are needed!\\n")
`,
    learningObjectives: [
      'Define individual and average causal effects using potential outcomes',
      'Understand the fundamental problem of causal inference',
      'Recognize that causal effects are contrasts of potential outcomes'
    ]
  },

  {
    id: 'randomization-exchangeability',
    title: 'Randomization and Exchangeability (What If Ch.2)',
    description: 'Why randomization solves the causal inference problem',
    category: 'theory',
    tier: 'Foundational',
    pythonCode: `# Randomization creates Exchangeability
# Based on Hernán & Robins "What If" Chapter 2

import numpy as np
import pandas as pd

np.random.seed(123)
n = 2000

# Generate baseline covariate (e.g., age)
age = np.random.normal(50, 15, n)

# Generate potential outcomes that depend on age
# Older people have worse outcomes
Y1 = np.random.binomial(1, 0.7 - 0.005 * (age - 50), n)
Y0 = np.random.binomial(1, 0.5 - 0.005 * (age - 50), n)

true_ate = np.mean(Y1 - Y0)

print("=== RANDOMIZATION AND EXCHANGEABILITY ===")
print(f"True ATE: {true_ate:.3f}\\n")

# SCENARIO 1: Observational study with selection bias
# Older people more likely to receive treatment
prob_treatment = 0.3 + 0.01 * (age - 50)
prob_treatment = np.clip(prob_treatment, 0.05, 0.95)
A_obs = np.random.binomial(1, prob_treatment, n)

# Observed outcomes (consistency: Y = Y^A)
Y_obs = A_obs * Y1 + (1 - A_obs) * Y0

# Naive estimate (association, not causation!)
naive_estimate = np.mean(Y_obs[A_obs==1]) - np.mean(Y_obs[A_obs==0])

print("OBSERVATIONAL STUDY (selection bias):")
print(f"  Mean age in treated: {np.mean(age[A_obs==1]):.1f}")
print(f"  Mean age in untreated: {np.mean(age[A_obs==0]):.1f}")
print(f"  Naive estimate: {naive_estimate:.3f}")
print(f"  Bias: {naive_estimate - true_ate:.3f}")
print(f"  ❌ Treated and untreated are NOT exchangeable\\n")

# SCENARIO 2: Randomized trial
# Treatment assigned randomly (50% probability for all)
A_rct = np.random.binomial(1, 0.5, n)
Y_rct = A_rct * Y1 + (1 - A_rct) * Y0

# Estimate from RCT
rct_estimate = np.mean(Y_rct[A_rct==1]) - np.mean(Y_rct[A_rct==0])

print("RANDOMIZED CONTROLLED TRIAL:")
print(f"  Mean age in treated: {np.mean(age[A_rct==1]):.1f}")
print(f"  Mean age in untreated: {np.mean(age[A_rct==0]):.1f}")
print(f"  RCT estimate: {rct_estimate:.3f}")
print(f"  Bias: {rct_estimate - true_ate:.3f}")
print(f"  ✓ Treated and untreated ARE exchangeable")

print("\\nKEY INSIGHT:")
print("Exchangeability means: Y^a ⊥ A for all a")
print("In words: potential outcomes are independent of treatment received")
print("Randomization ensures exchangeability, eliminating confounding!")
`,
    rCode: `# Randomization creates Exchangeability
# Based on Hernán & Robins "What If" Chapter 2

set.seed(123)
n <- 2000

# Generate baseline covariate
age <- rnorm(n, 50, 15)

# Generate potential outcomes that depend on age
Y1 <- rbinom(n, 1, pmax(0.01, pmin(0.99, 0.7 - 0.005 * (age - 50))))
Y0 <- rbinom(n, 1, pmax(0.01, pmin(0.99, 0.5 - 0.005 * (age - 50))))

true_ate <- mean(Y1 - Y0)

cat("=== RANDOMIZATION AND EXCHANGEABILITY ===\\n")
cat("True ATE:", round(true_ate, 3), "\\n\\n")

# SCENARIO 1: Observational study
prob_treatment <- pmax(0.05, pmin(0.95, 0.3 + 0.01 * (age - 50)))
A_obs <- rbinom(n, 1, prob_treatment)
Y_obs <- A_obs * Y1 + (1 - A_obs) * Y0

naive_estimate <- mean(Y_obs[A_obs==1]) - mean(Y_obs[A_obs==0])

cat("OBSERVATIONAL STUDY (selection bias):\\n")
cat("  Mean age in treated:", round(mean(age[A_obs==1]), 1), "\\n")
cat("  Mean age in untreated:", round(mean(age[A_obs==0]), 1), "\\n")
cat("  Naive estimate:", round(naive_estimate, 3), "\\n")
cat("  Bias:", round(naive_estimate - true_ate, 3), "\\n")
cat("  ❌ Treated and untreated are NOT exchangeable\\n\\n")

# SCENARIO 2: RCT
A_rct <- rbinom(n, 1, 0.5)
Y_rct <- A_rct * Y1 + (1 - A_rct) * Y0

rct_estimate <- mean(Y_rct[A_rct==1]) - mean(Y_rct[A_rct==0])

cat("RANDOMIZED CONTROLLED TRIAL:\\n")
cat("  Mean age in treated:", round(mean(age[A_rct==1]), 1), "\\n")
cat("  Mean age in untreated:", round(mean(age[A_rct==0]), 1), "\\n")
cat("  RCT estimate:", round(rct_estimate, 3), "\\n")
cat("  Bias:", round(rct_estimate - true_ate, 3), "\\n")
cat("  ✓ Treated and untreated ARE exchangeable\\n")

cat("\\nKEY INSIGHT:\\n")
cat("Exchangeability means: Y^a ⊥ A for all a\\n")
cat("Randomization ensures exchangeability!\\n")
`,
    learningObjectives: [
      'Understand the concept of exchangeability',
      'Recognize how randomization creates exchangeability',
      'Distinguish between association and causation'
    ]
  },

  // LESSON FROM PEARL'S "CAUSALITY"
  {
    id: 'do-calculus-backdoor',
    title: 'do-Calculus and Backdoor Criterion (Pearl Ch.3)',
    description: 'Pearl\'s do-operator and backdoor adjustment formula',
    category: 'theory',
    tier: 'Intermediate',
    pythonCode: `# do-Calculus: Backdoor Criterion
# Based on Pearl "Causality" Chapter 3

import numpy as np
import pandas as pd

np.random.seed(456)
n = 3000

# Simulate a simple DAG: Z → X → Y and Z → Y (Z is confounder)
# Z: baseline health (higher is better)
Z = np.random.normal(100, 15, n)

# X: treatment (affected by Z)
X = np.random.binomial(1, 1/(1 + np.exp(-(Z - 100)/10)), n)

# Y: outcome (affected by both Z and X)
# True causal effect of X on Y is β = 10
beta_true = 10
Y = 50 + beta_true * X + 0.5 * Z + np.random.normal(0, 5, n)

print("=== DO-CALCULUS: BACKDOOR CRITERION ===")
print("DAG: Z → X → Y and Z → Y")
print("Goal: Estimate causal effect P(Y | do(X=1)) - P(Y | do(X=0))\\n")

# WRONG: Unadjusted estimate (confounded)
y1_naive = np.mean(Y[X==1])
y0_naive = np.mean(Y[X==0])
naive_effect = y1_naive - y0_naive

print("1. NAIVE (CONFOUNDED) ESTIMATE:")
print(f"   E[Y | X=1] = {y1_naive:.2f}")
print(f"   E[Y | X=0] = {y0_naive:.2f}")
print(f"   Naive effect: {naive_effect:.2f}")
print(f"   ❌ This is NOT the causal effect!\\n")

# CORRECT: Backdoor adjustment using Z
# P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)
# This blocks the backdoor path X ← Z → Y

# Stratify by Z (discretize into quintiles)
z_bins = pd.qcut(Z, q=5, labels=False, duplicates='drop')

# Backdoor adjustment formula
y1_adjusted = 0
y0_adjusted = 0

for z_val in np.unique(z_bins):
    mask = z_bins == z_val
    p_z = np.mean(mask)

    y1_z = np.mean(Y[(X==1) & mask]) if np.any((X==1) & mask) else 0
    y0_z = np.mean(Y[(X==0) & mask]) if np.any((X==0) & mask) else 0

    y1_adjusted += y1_z * p_z
    y0_adjusted += y0_z * p_z

backdoor_effect = y1_adjusted - y0_adjusted

print("2. BACKDOOR ADJUSTMENT (CORRECT):")
print(f"   E[Y | do(X=1)] = Σ_z E[Y | X=1, Z=z] P(Z=z) = {y1_adjusted:.2f}")
print(f"   E[Y | do(X=0)] = Σ_z E[Y | X=0, Z=z] P(Z=z) = {y0_adjusted:.2f}")
print(f"   Causal effect: {backdoor_effect:.2f}")
print(f"   True effect: {beta_true:.2f}")
print(f"   ✓ Backdoor criterion satisfied!\\n")

# Alternative: Regression adjustment
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(np.column_stack([X, Z]), Y)
regression_effect = model.coef_[0]

print("3. REGRESSION ADJUSTMENT (EQUIVALENT):")
print(f"   Coefficient on X: {regression_effect:.2f}")
print(f"   ✓ Adjusting for Z blocks backdoor path\\n")

print("KEY INSIGHTS:")
print("- do(X=x) represents an intervention, setting X to x")
print("- Backdoor criterion: adjust for Z to block non-causal paths")
print("- Adjustment formula: E[Y|do(X)] = Σ_z E[Y|X,Z=z]P(Z=z)")
`,
    rCode: `# do-Calculus: Backdoor Criterion
# Based on Pearl "Causality" Chapter 3

set.seed(456)
n <- 3000

# DAG: Z → X → Y and Z → Y
Z <- rnorm(n, 100, 15)
X <- rbinom(n, 1, plogis((Z - 100)/10))

beta_true <- 10
Y <- 50 + beta_true * X + 0.5 * Z + rnorm(n, 0, 5)

cat("=== DO-CALCULUS: BACKDOOR CRITERION ===\\n")
cat("DAG: Z → X → Y and Z → Y\\n")
cat("Goal: Estimate P(Y | do(X=1)) - P(Y | do(X=0))\\n\\n")

# NAIVE
y1_naive <- mean(Y[X==1])
y0_naive <- mean(Y[X==0])
naive_effect <- y1_naive - y0_naive

cat("1. NAIVE (CONFOUNDED) ESTIMATE:\\n")
cat("   E[Y | X=1] =", round(y1_naive, 2), "\\n")
cat("   E[Y | X=0] =", round(y0_naive, 2), "\\n")
cat("   Naive effect:", round(naive_effect, 2), "\\n")
cat("   ❌ This is NOT the causal effect!\\n\\n")

# BACKDOOR ADJUSTMENT
z_bins <- as.numeric(cut(Z, breaks=quantile(Z, probs=seq(0,1,0.2)), include.lowest=TRUE))

y1_adjusted <- 0
y0_adjusted <- 0

for (z_val in unique(z_bins)) {
  mask <- z_bins == z_val
  p_z <- mean(mask)

  y1_z <- ifelse(sum(X==1 & mask) > 0, mean(Y[X==1 & mask]), 0)
  y0_z <- ifelse(sum(X==0 & mask) > 0, mean(Y[X==0 & mask]), 0)

  y1_adjusted <- y1_adjusted + y1_z * p_z
  y0_adjusted <- y0_adjusted + y0_z * p_z
}

backdoor_effect <- y1_adjusted - y0_adjusted

cat("2. BACKDOOR ADJUSTMENT (CORRECT):\\n")
cat("   E[Y | do(X=1)] =", round(y1_adjusted, 2), "\\n")
cat("   E[Y | do(X=0)] =", round(y0_adjusted, 2), "\\n")
cat("   Causal effect:", round(backdoor_effect, 2), "\\n")
cat("   True effect:", beta_true, "\\n")
cat("   ✓ Backdoor criterion satisfied!\\n\\n")

# REGRESSION
model <- lm(Y ~ X + Z)
regression_effect <- coef(model)["X"]

cat("3. REGRESSION ADJUSTMENT:\\n")
cat("   Coefficient on X:", round(regression_effect, 2), "\\n")
cat("   ✓ Adjusting for Z blocks backdoor path\\n")
`,
    learningObjectives: [
      'Understand Pearl\'s do-operator and interventions',
      'Apply the backdoor criterion for confounding adjustment',
      'Implement backdoor adjustment formula'
    ]
  },

  {
    id: 'ipw-what-if-ch12',
    title: 'Inverse Probability Weighting (What If Ch.12)',
    description: 'IPW estimator for marginal structural models',
    category: 'theory',
    tier: 'Intermediate',
    pythonCode: `# Inverse Probability Weighting (IPW)
# Based on Hernán & Robins "What If" Chapter 12

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

np.random.seed(789)
n = 5000

# Generate data with confounding
L = np.random.normal(0, 1, n)  # Confounder
A = np.random.binomial(1, 1/(1 + np.exp(-L)), n)  # Treatment depends on L
Y = 2*A + L + np.random.normal(0, 1, n)  # Outcome depends on both

true_ate = 2.0

print("=== INVERSE PROBABILITY WEIGHTING (IPW) ===")
print(f"True ATE: {true_ate}\\n")

# Step 1: Estimate propensity score P(A=1|L)
ps_model = LogisticRegression()
ps_model.fit(L.reshape(-1, 1), A)
ps = ps_model.predict_proba(L.reshape(-1, 1))[:, 1]

print("STEP 1: Estimate propensity scores")
print(f"  Propensity score range: [{ps.min():.3f}, {ps.max():.3f}]")
print(f"  Mean PS in treated: {ps[A==1].mean():.3f}")
print(f"  Mean PS in untreated: {ps[A==0].mean():.3f}\\n")

# Step 2: Compute IPW weights
# Weight = 1/P(A|L) for treated, 1/(1-P(A|L)) for untreated
weights = np.where(A == 1, 1/ps, 1/(1-ps))

print("STEP 2: Compute IPW weights")
print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
print(f"  Mean weight in treated: {weights[A==1].mean():.2f}")
print(f"  Mean weight in untreated: {weights[A==0].mean():.2f}\\n")

# Check for extreme weights (positivity violations)
if weights.max() > 100:
    print("  ⚠️ WARNING: Extreme weights detected (possible positivity violations)")
else:
    print("  ✓ Weights look reasonable\\n")

# Step 3: Estimate ATE using weighted means
# E[Y^1] = E[Y * A / P(A|L)] / E[A / P(A|L)]
# E[Y^0] = E[Y * (1-A) / (1-P(A|L))] / E[(1-A) / (1-P(A|L))]

y1_ipw = np.sum(Y * A * weights) / np.sum(A * weights)
y0_ipw = np.sum(Y * (1-A) * weights) / np.sum((1-A) * weights)
ate_ipw = y1_ipw - y0_ipw

print("STEP 3: IPW Estimate of ATE")
print(f"  E[Y^1] (weighted) = {y1_ipw:.3f}")
print(f"  E[Y^0] (weighted) = {y0_ipw:.3f}")
print(f"  ATE (IPW) = {ate_ipw:.3f}")
print(f"  True ATE = {true_ate:.3f}")
print(f"  Bias = {abs(ate_ipw - true_ate):.3f}\\n")

# Stabilized weights (for comparison)
p_a = np.mean(A)
sw = np.where(A == 1, p_a/ps, (1-p_a)/(1-ps))

ate_sw = np.sum(Y * A * sw) / np.sum(A * sw) - np.sum(Y * (1-A) * sw) / np.sum((1-A) * sw)

print("STABILIZED WEIGHTS (variance reduction):")
print(f"  Stabilized weight range: [{sw.min():.2f}, {sw.max():.2f}]")
print(f"  ATE (stabilized IPW) = {ate_sw:.3f}\\n")

print("KEY INSIGHTS:")
print("- IPW creates a pseudo-population where A ⊥ L")
print("- Weights inversely proportional to probability of receiving actual treatment")
print("- Stabilized weights: numerator = P(A), denominator = P(A|L)")
print("- Extreme weights indicate near-violations of positivity")
`,
    rCode: `# Inverse Probability Weighting (IPW)
# Based on Hernán & Robins "What If" Chapter 12

set.seed(789)
n <- 5000

L <- rnorm(n, 0, 1)
A <- rbinom(n, 1, plogis(L))
Y <- 2*A + L + rnorm(n, 0, 1)

true_ate <- 2.0

cat("=== INVERSE PROBABILITY WEIGHTING (IPW) ===\\n")
cat("True ATE:", true_ate, "\\n\\n")

# Estimate propensity scores
ps_model <- glm(A ~ L, family=binomial)
ps <- predict(ps_model, type="response")

cat("STEP 1: Estimate propensity scores\\n")
cat("  PS range: [", round(min(ps), 3), ",", round(max(ps), 3), "]\\n")
cat("  Mean PS in treated:", round(mean(ps[A==1]), 3), "\\n")
cat("  Mean PS in untreated:", round(mean(ps[A==0]), 3), "\\n\\n")

# IPW weights
weights <- ifelse(A == 1, 1/ps, 1/(1-ps))

cat("STEP 2: Compute IPW weights\\n")
cat("  Weight range: [", round(min(weights), 2), ",", round(max(weights), 2), "]\\n")
cat("  Mean weight in treated:", round(mean(weights[A==1]), 2), "\\n")
cat("  Mean weight in untreated:", round(mean(weights[A==0]), 2), "\\n\\n")

# Estimate ATE
y1_ipw <- sum(Y * A * weights) / sum(A * weights)
y0_ipw <- sum(Y * (1-A) * weights) / sum((1-A) * weights)
ate_ipw <- y1_ipw - y0_ipw

cat("STEP 3: IPW Estimate\\n")
cat("  E[Y^1] (weighted) =", round(y1_ipw, 3), "\\n")
cat("  E[Y^0] (weighted) =", round(y0_ipw, 3), "\\n")
cat("  ATE (IPW) =", round(ate_ipw, 3), "\\n")
cat("  True ATE =", true_ate, "\\n")
cat("  Bias =", round(abs(ate_ipw - true_ate), 3), "\\n\\n")

# Stabilized weights
p_a <- mean(A)
sw <- ifelse(A == 1, p_a/ps, (1-p_a)/(1-ps))
ate_sw <- sum(Y * A * sw) / sum(A * sw) - sum(Y * (1-A) * sw) / sum((1-A) * sw)

cat("STABILIZED WEIGHTS:\\n")
cat("  SW range: [", round(min(sw), 2), ",", round(max(sw), 2), "]\\n")
cat("  ATE (stabilized) =", round(ate_sw, 3), "\\n")
`,
    learningObjectives: [
      'Compute and interpret propensity scores',
      'Calculate IPW weights for causal estimation',
      'Understand stabilized weights and positivity checks'
    ]
  },
];

// Export all comprehensive lessons
export default comprehensiveLessons;
