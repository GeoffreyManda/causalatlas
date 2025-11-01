export interface TheoryTopic {
  id: string;
  title: string;
  tier: 'Foundational' | 'Intermediate' | 'Advanced';
  description: string;
  content: string;
  prerequisites: string[];
  learningObjectives: string[];
  keyDefinitions: { term: string; definition: string }[];
  examples: {
    python: string;
    r: string;
  };
  references: {
    authors: string;
    title: string;
    year: number;
    doi: string;
  }[];
}

export const causalTheory: TheoryTopic[] = [
  // ========== FOUNDATIONAL ==========
  {
    id: 'intro_causal_inference',
    title: 'Introduction to Causal Inference',
    tier: 'Foundational',
    description: 'From causality fundamentals to estimands, estimators, and estimates',
    content: `## What is Causality?

Causality answers "what if" questions about interventions. Unlike statistical association (correlation), causality describes what happens when we actively change a variable.

**Key Distinction:**
- **Association**: X and Y occur together → P(Y|X)
- **Causation**: Changing X causes Y to change → P(Y|do(X))

The **do-operator** represents an intervention that sets X to a value, breaking all incoming arrows to X in a causal graph.

**Example: Confounding**
Ice cream sales and drowning deaths are correlated, but ice cream doesn't cause drowning. Both are caused by hot weather (the confounder). The association is spurious.

---

## The Causal Inference Framework

Causal inference is built on three fundamental concepts:

**1. Estimand** - The target quantity we want to know
- A well-defined causal parameter (e.g., average treatment effect)
- Represents the "truth" we seek in the population
- Independent of any particular study or data
- Example: E[Y¹ - Y⁰] = average causal effect of treatment

**2. Estimator** - The method/formula to compute an estimate
- A statistical procedure or algorithm
- Examples: regression adjustment, inverse probability weighting, AIPW, TMLE
- Different estimators can target the same estimand
- Quality: bias, variance, robustness

**3. Estimate** - The numerical answer from applying an estimator to data
- A specific number (e.g., ATE = 2.3 units)
- Subject to sampling variability and bias
- Quality depends on both the estimator and the data

---

## The Causal Inference Pipeline

**Define Estimand → Choose Valid Estimator → Obtain Estimate → Quantify Uncertainty**

**Why This Matters:**
- **Clarity** in defining what we're trying to learn (estimand)
- **Transparency** in how we compute it (estimator)
- **Honesty** about what the data tells us (estimate)
- **Reproducibility** through explicit formalization`,
    prerequisites: [],
    learningObjectives: [
      'Distinguish causation from association',
      'Understand the do-operator and confounding',
      'Distinguish estimands, estimators, and estimates',
      'Understand the causal inference workflow'
    ],
    keyDefinitions: [
      { term: 'Intervention', definition: 'An external action that sets a variable to a specific value' },
      { term: 'do-operator', definition: 'do(X=x) represents intervention, written P(Y|do(X=x))' },
      { term: 'Confounding', definition: 'A common cause of treatment and outcome that creates spurious association' },
      { term: 'Estimand', definition: 'The causal parameter of interest (the target)' },
      { term: 'Estimator', definition: 'The statistical method to estimate the estimand' },
      { term: 'Estimate', definition: 'The numerical result from applying an estimator to data' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 1000

# ========== PART 1: Confounding ==========
print("=== CONFOUNDING EXAMPLE ===")

# Confounder Z (e.g., age)
Z = np.random.normal(50, 15, n)

# Treatment X depends on Z
X = (Z > 50).astype(int) + np.random.binomial(1, 0.1, n)
X = np.clip(X, 0, 1)

# Outcome Y depends on Z, NOT on X
Y_conf = 2 * Z + np.random.normal(0, 10, n)

# Naive association (WRONG!)
assoc = Y_conf[X==1].mean() - Y_conf[X==0].mean()
print(f"Naive association: {assoc:.2f}")
print("^ Shows 'effect' but X doesn't cause Y!")

# Causal effect (controlling for Z)
model = LinearRegression().fit(np.c_[X, Z], Y_conf)
print(f"Causal effect (controlling Z): {model.coef_[0]:.2f}")
print("^ Close to 0: no causal relationship\\n")

# ========== PART 2: Estimand/Estimator/Estimate ==========
print("=== ESTIMAND/ESTIMATOR/ESTIMATE ===")

# ESTIMAND: True ATE in population
true_ATE = 2.0
print(f"TRUE ESTIMAND (ATE): {true_ATE}")

# Generate data with true causal effect
A = np.random.binomial(1, 0.5, n)
Y = true_ATE * A + np.random.normal(0, 1, n)

# ESTIMATOR 1: Difference in means
est1 = Y[A==1].mean() - Y[A==0].mean()

# ESTIMATOR 2: Regression
est2 = LinearRegression().fit(A.reshape(-1,1), Y).coef_[0]

# ESTIMATES: What we observe
print(f"Estimate 1 (diff-in-means): {est1:.3f}")
print(f"Estimate 2 (regression): {est2:.3f}")
print("→ Different estimators, same estimand, similar estimates")`,
      r: `set.seed(42)
n <- 1000

# ========== PART 1: Confounding ==========
cat("=== CONFOUNDING EXAMPLE ===\\n")

# Confounder Z
Z <- rnorm(n, 50, 15)

# Treatment X depends on Z
X <- as.integer(Z > 50) + rbinom(n, 1, 0.1)
X <- pmin(X, 1)

# Outcome Y depends on Z, NOT on X
Y_conf <- 2 * Z + rnorm(n, 0, 10)

# Naive association
cat("Naive association:", round(mean(Y_conf[X==1]) - mean(Y_conf[X==0]), 2), "\\n")

# Causal effect (controlling Z)
model <- lm(Y_conf ~ X + Z)
cat("Causal effect (controlling Z):", round(coef(model)[2], 2), "\\n\\n")

# ========== PART 2: Estimand/Estimator/Estimate ==========
cat("=== ESTIMAND/ESTIMATOR/ESTIMATE ===\\n")

# ESTIMAND: True ATE
true_ATE <- 2.0
cat("TRUE ESTIMAND (ATE):", true_ATE, "\\n")

# Generate data
A <- rbinom(n, 1, 0.5)
Y <- true_ATE * A + rnorm(n)

# ESTIMATOR 1: Difference in means
est1 <- mean(Y[A==1]) - mean(Y[A==0])

# ESTIMATOR 2: Regression
est2 <- coef(lm(Y ~ A))[2]

cat("Estimate 1 (diff-in-means):", round(est1, 3), "\\n")
cat("Estimate 2 (regression):", round(est2, 3), "\\n")`
    },
    references: [
      { authors: 'Pearl J', title: 'Causality: Models, Reasoning, and Inference', year: 2009, doi: '10.1017/CBO9780511803161' },
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If', year: 2020, doi: '10.1201/9781420013542' },
      { authors: 'Lundberg I et al', title: 'What is your estimand?', year: 2021, doi: '10.1177/0081175021101969' }
    ]
  },

  {
    id: 'dags_basics',
    title: 'Directed Acyclic Graphs (DAGs)',
    tier: 'Foundational',
    description: 'Visual representation of causal relationships using graphs',
    content: `A DAG encodes causal assumptions:
- **Nodes** = Variables
- **Directed edges** (arrows) = Direct causal effects  
- **Acyclic** = No feedback loops

**Key Structures:**
1. **Chain**: X → Z → Y (Z mediates X's effect on Y)
2. **Fork**: X ← Z → Y (Z confounds X and Y)
3. **Collider**: X → Z ← Y (Z is caused by both X and Y)

**Structural Equations:**
Each node has equation: V := f_V(Parents_V, U_V) where U_V is noise.`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Draw DAGs for causal scenarios',
      'Identify chains, forks, colliders',
      'Write structural equations from DAGs'
    ],
    keyDefinitions: [
      { term: 'DAG', definition: 'Directed Acyclic Graph - nodes and arrows with no cycles' },
      { term: 'Parent', definition: 'A node with arrow pointing to another node' },
      { term: 'Structural Equation', definition: 'Equation defining how each variable is generated' }
    ],
    examples: {
      python: `import numpy as np

# Simulate DAG: X → Z → Y (chain)
np.random.seed(42)
n = 2000

# Structural equations
X = np.random.normal(0, 1, n)
Z = 0.6 * X + np.random.normal(0, 1, n)  # Z caused by X
Y = 0.8 * Z + np.random.normal(0, 1, n)  # Y caused by Z

# Total effect of X on Y = 0.6 * 0.8 = 0.48
from sklearn.linear_model import LinearRegression

# Total effect (X → Y)
total_effect = LinearRegression().fit(X.reshape(-1,1), Y).coef_[0]
print(f"Total effect X → Y: {total_effect:.3f}")
print(f"Expected: {0.6 * 0.8:.3f}")

# Direct effect blocked when conditioning on Z
model_full = LinearRegression().fit(np.c_[X, Z], Y)
print(f"\\nDirect effect X → Y | Z: {model_full.coef_[0]:.3f}")
print("^ ~0 because X only affects Y through Z")`,
      r: `set.seed(42)
n <- 2000

# Structural equations
X <- rnorm(n)
Z <- 0.6 * X + rnorm(n)
Y <- 0.8 * Z + rnorm(n)

# Total effect
total <- coef(lm(Y ~ X))[2]
cat("Total effect X → Y:", round(total, 3), "\\n")
cat("Expected:", 0.6 * 0.8, "\\n")

# Direct effect
model <- lm(Y ~ X + Z)
cat("\\nDirect effect X → Y | Z:", round(coef(model)[2], 3), "\\n")`
    },
    references: [
      { authors: 'Pearl J', title: 'Causal diagrams for empirical research', year: 1995, doi: '10.2307/2337329' }
    ]
  },

  {
    id: 'd_separation',
    title: 'd-separation',
    tier: 'Intermediate',
    description: 'Graph-based criterion for conditional independence',
    content: `d-separation determines if two variables are independent given a conditioning set.

**Blocking Rules:**
1. **Chain** (X → Z → Y): Z blocks path
2. **Fork** (X ← Z → Y): Z blocks path  
3. **Collider** (X → Z ← Y): Z OPENS path (blocked without Z)

**Definition:** X and Y are d-separated by Z if all paths between X and Y are blocked by Z.

**Implication:** d-sep(X,Y|Z) in DAG → X ⊥ Y | Z in probability`,
    prerequisites: ['dags_basics'],
    learningObjectives: [
      'Apply blocking rules to paths',
      'Determine d-separation from DAG',
      'Identify sufficient adjustment sets'
    ],
    keyDefinitions: [
      { term: 'd-separation', definition: 'Graph criterion that implies conditional independence' },
      { term: 'Blocked path', definition: 'Path that cannot transmit association' },
      { term: 'Collider bias', definition: 'Conditioning on collider creates spurious association' }
    ],
    examples: {
      python: `import numpy as np
from scipy.stats import chi2_contingency

np.random.seed(42)
n = 5000

# Fork: X ← Z → Y (Z is confounder)
Z = np.random.binomial(1, 0.5, n)
X = np.random.binomial(1, 0.3 + 0.4*Z, n)
Y = np.random.binomial(1, 0.3 + 0.4*Z, n)

# X and Y dependent unconditionally
table = np.array([[((X==0) & (Y==0)).sum(), ((X==0) & (Y==1)).sum()],
                  [((X==1) & (Y==0)).sum(), ((X==1) & (Y==1)).sum()]])
chi2, p, _, _ = chi2_contingency(table)
print(f"X ⊥ Y? p={p:.4f} (dependent, path open)")

# X and Y independent given Z
for z in [0, 1]:
    mask = Z == z
    table_z = np.array([[((X==0) & (Y==0) & mask).sum(), ((X==0) & (Y==1) & mask).sum()],
                        [((X==1) & (Y==0) & mask).sum(), ((X==1) & (Y==1) & mask).sum()]])
    _, p_z, _, _ = chi2_contingency(table_z + 0.5)  # Add 0.5 for stability
    print(f"X ⊥ Y | Z={z}? p={p_z:.4f} (independent, path blocked)")`,
      r: `set.seed(42)
n <- 5000

# Fork: X ← Z → Y
Z <- rbinom(n, 1, 0.5)
X <- rbinom(n, 1, 0.3 + 0.4*Z)
Y <- rbinom(n, 1, 0.3 + 0.4*Z)

# Unconditional test
test_uncond <- chisq.test(table(X, Y))
cat("X ⊥ Y? p=", round(test_uncond$p.value, 4), "\\n")

# Conditional tests
for(z in 0:1) {
  test_cond <- chisq.test(table(X[Z==z], Y[Z==z]))
  cat("X ⊥ Y | Z=", z, "? p=", round(test_cond$p.value, 4), "\\n")
}`
    },
    references: [
      { authors: 'Pearl J', title: 'Probabilistic reasoning using graphs', year: 1988, doi: '10.1145/1056126.1056128' }
    ]
  },

  {
    id: 'do_calculus',
    title: 'do-Calculus',
    tier: 'Intermediate',
    description: 'Three rules for transforming interventional distributions',
    content: `do-Calculus provides algebraic rules to identify causal effects from observational data.

**Three Rules:**
1. **Insertion/deletion of observations**: P(y|do(x),z,w) = P(y|do(x),w) if (Y ⊥ Z | X,W) in graph with arrows into X removed
2. **Action/observation exchange**: P(y|do(x),do(z),w) = P(y|do(x),z,w) if (Y ⊥ Z | X,W) in graph with arrows into X removed and outgoing from Z removed  
3. **Insertion/deletion of actions**: P(y|do(x),do(z),w) = P(y|do(x),w) if (Y ⊥ Z | X,W) in graph with arrows into X removed and incoming to Z removed

These rules + graphical criteria (backdoor, frontdoor) enable identification.`,
    prerequisites: ['d_separation'],
    learningObjectives: [
      'Apply backdoor criterion',
      'Apply frontdoor criterion', 
      'Use do-calculus to derive identification formulas'
    ],
    keyDefinitions: [
      { term: 'do-Calculus', definition: 'Three rules for manipulating do-expressions' },
      { term: 'Backdoor criterion', definition: 'Condition for identifying causal effects by adjustment' },
      { term: 'Frontdoor criterion', definition: 'Identifies effects through mediators when backdoor fails' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 3000

# DAG: X ← Z → Y, X → Y (backdoor Z)
Z = np.random.normal(0, 1, n)
X = 0.6*Z + np.random.normal(0, 1, n)
Y = 1.5*X + 0.8*Z + np.random.normal(0, 1, n)

# Backdoor adjustment formula:
# P(y|do(x)) = sum_z P(y|x,z) P(z)

# Implementation via regression
model = LinearRegression().fit(np.c_[X, Z], Y)
causal_effect = model.coef_[0]

print(f"Causal effect via backdoor adjustment: {causal_effect:.3f}")
print(f"True causal effect: 1.500")

# Compare to biased estimate without adjustment
biased = LinearRegression().fit(X.reshape(-1,1), Y).coef_[0]
print(f"\\nBiased estimate (no adjustment): {biased:.3f}")
print("^ Biased due to confounding by Z")`,
      r: `set.seed(42)
n <- 3000

# DAG: X ← Z → Y, X → Y
Z <- rnorm(n)
X <- 0.6*Z + rnorm(n)
Y <- 1.5*X + 0.8*Z + rnorm(n)

# Backdoor adjustment
model <- lm(Y ~ X + Z)
causal <- coef(model)[2]
cat("Causal effect via backdoor:", round(causal, 3), "\\n")

# Biased without adjustment
biased <- coef(lm(Y ~ X))[2]
cat("Biased estimate:", round(biased, 3), "\\n")`
    },
    references: [
      { authors: 'Pearl J', title: 'Causal diagrams for empirical research', year: 1995, doi: '10.2307/2337329' }
    ]
  },

  {
    id: 'graphoids',
    title: 'Graphoid Axioms',
    tier: 'Advanced',
    description: 'Axiomatic properties of conditional independence',
    content: `Graphoid axioms are properties that conditional independence relations satisfy.

**Semi-Graphoid Axioms:**
1. **Symmetry**: (X ⊥ Y | Z) ⟹ (Y ⊥ X | Z)
2. **Decomposition**: (X ⊥ Y,W | Z) ⟹ (X ⊥ Y | Z)
3. **Weak Union**: (X ⊥ Y,W | Z) ⟹ (X ⊥ Y | Z,W)
4. **Contraction**: (X ⊥ Y | Z) ∧ (X ⊥ W | Z,Y) ⟹ (X ⊥ Y,W | Z)

**Full Graphoid adds:**
5. **Intersection**: (X ⊥ W | Z,Y) ∧ (X ⊥ Y | Z,W) ⟹ (X ⊥ Y,W | Z)

These axioms enable reasoning about d-separation without checking all paths.`,
    prerequisites: ['d_separation'],
    learningObjectives: [
      'Verify graphoid axioms',
      'Use axioms to derive new independencies',
      'Understand when intersection fails'
    ],
    keyDefinitions: [
      { term: 'Semi-graphoid', definition: 'System satisfying symmetry, decomposition, weak union, contraction' },
      { term: 'Graphoid', definition: 'Semi-graphoid plus intersection property' },
      { term: 'Faithful', definition: 'DAG where d-separation implies all independencies' }
    ],
    examples: {
      python: `import numpy as np

# Verify symmetry axiom
np.random.seed(42)
n = 10000

Z = np.random.normal(0, 1, n)
X = Z + np.random.normal(0, 1, n)
Y = Z + np.random.normal(0, 1, n)

# Partial correlations
def partial_corr(x, y, z):
    from scipy.stats import pearsonr
    res_x = x - np.polyfit(z, x, 1)[0]*z
    res_y = y - np.polyfit(z, y, 1)[0]*z
    return pearsonr(res_x, res_y)[0]

r_xy_z = partial_corr(X, Y, Z)
r_yx_z = partial_corr(Y, X, Z)

print("Symmetry: X ⊥ Y | Z ⟺ Y ⊥ X | Z")
print(f"  Partial corr(X,Y|Z): {r_xy_z:.4f}")
print(f"  Partial corr(Y,X|Z): {r_yx_z:.4f}")
print(f"  Equal? {abs(r_xy_z - r_yx_z) < 0.01}")`,
      r: `library(ppcor)

set.seed(42)
n <- 10000

Z <- rnorm(n)
X <- Z + rnorm(n)
Y <- Z + rnorm(n)

# Partial correlations
r_xy_z <- pcor.test(X, Y, Z)$estimate
r_yx_z <- pcor.test(Y, X, Z)$estimate

cat("Symmetry axiom:\\n")
cat("  Partial corr(X,Y|Z):", round(r_xy_z, 4), "\\n")
cat("  Partial corr(Y,X|Z):", round(r_yx_z, 4), "\\n")`
    },
    references: [
      { authors: 'Pearl J, Paz A', title: 'Graphoids: graph-based logic', year: 1987, doi: '10.1016/0304-3975(87)90136-4' }
    ]
  },

  {
    id: 'identification',
    title: 'Causal Identification',
    tier: 'Advanced',
    description: 'Determining if causal effects can be estimated from observational data',
    content: `Identification asks: Can we express P(Y|do(X)) using only observed distributions?

**Main Results:**
- **Backdoor Criterion**: If Z blocks all backdoor paths X ↝ Y, then P(y|do(x)) = Σ_z P(y|x,z)P(z)
- **Frontdoor Criterion**: If M fully mediates X → Y and has no backdoor paths, identification possible
- **ID Algorithm**: Complete algorithm to determine if any causal effect is identifiable

**Non-Identification:** When unmeasured confounding exists and no valid adjustment set exists, bounds may be derived instead.`,
    prerequisites: ['do_calculus'],
    learningObjectives: [
      'Check backdoor criterion',
      'Apply ID algorithm concepts',
      'Recognize non-identifiable cases'
    ],
    keyDefinitions: [
      { term: 'Identifiable', definition: 'Causal effect expressible from observational distribution' },
      { term: 'Backdoor path', definition: 'Path X ← ... → Y that creates confounding' },
      { term: 'ID Algorithm', definition: 'Complete procedure for determining identifiability' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression

# Case 1: Identifiable (backdoor adjustment exists)
np.random.seed(42)
n = 2000

Z = np.random.normal(0, 1, n)  # Observed confounder
X = 0.5*Z + np.random.normal(0, 1, n)
Y = 1.8*X + 0.7*Z + np.random.normal(0, 1, n)

# Backdoor adjustment
model = LinearRegression().fit(np.c_[X, Z], Y)
identified = model.coef_[0]
print(f"Case 1 - Identifiable effect: {identified:.3f}")
print("True effect: 1.800\\n")

# Case 2: Non-identifiable (unmeasured confounding)
U = np.random.normal(0, 1, n)  # UNMEASURED confounder
X2 = 0.5*U + np.random.normal(0, 1, n)
Y2 = 1.8*X2 + 0.7*U + np.random.normal(0, 1, n)

naive = LinearRegression().fit(X2.reshape(-1,1), Y2).coef_[0]
print(f"Case 2 - Non-identifiable (unmeasured U)")
print(f"  Naive estimate: {naive:.3f}")
print(f"  True effect: 1.800")
print("  Biased! Need bounds or sensitivity analysis")`,
      r: `set.seed(42)
n <- 2000

# Case 1: Identifiable
Z <- rnorm(n)
X <- 0.5*Z + rnorm(n)
Y <- 1.8*X + 0.7*Z + rnorm(n)

identified <- coef(lm(Y ~ X + Z))[2]
cat("Identifiable effect:", round(identified, 3), "\\n\\n")

# Case 2: Non-identifiable
U <- rnorm(n)
X2 <- 0.5*U + rnorm(n)
Y2 <- 1.8*X2 + 0.7*U + rnorm(n)

naive <- coef(lm(Y2 ~ X2))[2]
cat("Non-identifiable (unmeasured U)\\n")
cat("  Naive:", round(naive, 3), "\\n")
cat("  Biased! Need bounds\\n")`
    },
    references: [
      { authors: 'Shpitser I, Pearl J', title: 'Identification of joint interventional distributions', year: 2006, doi: '10.5555/1597348.1597382' }
    ]
  },

  // ========== FRAMEWORK-SPECIFIC THEORY ==========
  {
    id: 'framework_potential_outcomes',
    title: 'Potential Outcomes Framework',
    tier: 'Foundational',
    description: 'Rubin Causal Model and counterfactual reasoning',
    content: `The Potential Outcomes framework (Neyman-Rubin) represents causality through counterfactuals.

**Core Idea:**
Each unit has multiple potential outcomes Y^a for each treatment level a, but we only observe one.

**Fundamental Problem of Causal Inference:**
We can never observe Y^1 and Y^0 for the same unit simultaneously.

**SUTVA (Stable Unit Treatment Value Assumption):**
1. No interference between units
2. One version of each treatment

**Identification via Randomization:**
In RCTs: E[Y^a] = E[Y | A=a] because treatment assignment is independent of potential outcomes.

**Key Estimands:**
- ATE: E[Y^1 - Y^0]
- ATT: E[Y^1 - Y^0 | A=1]
- CATE: E[Y^1 - Y^0 | X=x]

**Compared to SCM:**
- PO focuses on "what-if" comparisons
- No explicit causal graph
- Exchangeability is central assumption`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Define potential outcomes',
      'Understand SUTVA',
      'Apply exchangeability for identification'
    ],
    keyDefinitions: [
      { term: 'Potential Outcome', definition: 'Y^a is outcome that would occur under treatment a' },
      { term: 'SUTVA', definition: 'Stable unit treatment value assumption' },
      { term: 'Exchangeability', definition: 'Y^a ⊥ A - treatment independent of potential outcomes' }
    ],
    examples: {
      python: `import numpy as np

np.random.seed(42)
n = 2000

# Generate BOTH potential outcomes (only possible in simulation)
Y0 = np.random.normal(10, 2, n)  # Potential outcome under control
Y1 = Y0 + 3 + np.random.normal(0, 1, n)  # Potential outcome under treatment

# True causal effect for each unit
individual_effect = Y1 - Y0
true_ATE = individual_effect.mean()

# Randomized treatment assignment
A = np.random.binomial(1, 0.5, n)

# Observed outcome (fundamental problem: only see one potential outcome)
Y_obs = np.where(A == 1, Y1, Y0)

# Estimate ATE
estimated_ATE = Y_obs[A==1].mean() - Y_obs[A==0].mean()

print(f"True ATE: {true_ATE:.3f}")
print(f"Estimated ATE: {estimated_ATE:.3f}")
print(f"\\nFundamental problem: We observe {(A==1).sum()} treated Y1 and {(A==0).sum()} control Y0")
print("But never both Y1 AND Y0 for the same person!")`,
      r: `set.seed(42)
n <- 2000

# Both potential outcomes
Y0 <- rnorm(n, 10, 2)
Y1 <- Y0 + 3 + rnorm(n)

# True effects
true_ATE <- mean(Y1 - Y0)

# Randomization
A <- rbinom(n, 1, 0.5)
Y_obs <- ifelse(A == 1, Y1, Y0)

# Estimate
est_ATE <- mean(Y_obs[A==1]) - mean(Y_obs[A==0])

cat("True ATE:", round(true_ATE, 3), "\\n")
cat("Estimated ATE:", round(est_ATE, 3), "\\n")`
    },
    references: [
      { authors: 'Rubin DB', title: 'Estimating causal effects of treatments', year: 1974, doi: '10.1037/h0037350' },
      { authors: 'Imbens GW, Rubin DB', title: 'Causal Inference for Statistics', year: 2015, doi: '10.1017/CBO9781139025751' }
    ]
  },

  {
    id: 'framework_scm',
    title: 'Structural Causal Models (SCM)',
    tier: 'Foundational',
    description: 'Pearl\'s graphical causal framework',
    content: `SCMs combine graphs and equations to represent causal mechanisms.

**SCM Components:**
1. **Variables** V = {X₁, ..., Xₙ}
2. **Structural equations** Xᵢ := fᵢ(Parentsᵢ, Uᵢ)
3. **DAG** showing direct causal effects
4. **Exogenous variables** U representing unmeasured factors

**The do-Operator:**
do(X=x) represents intervention:
- Sets X to value x
- Removes all incoming arrows to X
- P(Y | do(X=x)) ≠ P(Y | X=x) in general

**Identification:**
- Backdoor criterion
- Frontdoor criterion
- do-Calculus (3 rules)

**Compared to Potential Outcomes:**
- SCM makes causal structure explicit via DAGs
- do-operator vs potential outcomes notation
- Both are equivalent for point treatment effects

**Key Insight:**
Graphical criteria determine if causal effects are identifiable from observational data.`,
    prerequisites: ['dags_basics', 'do_calculus'],
    learningObjectives: [
      'Define structural causal models',
      'Use do-operator for interventions',
      'Apply graphical identification criteria'
    ],
    keyDefinitions: [
      { term: 'SCM', definition: 'Tuple (V, U, F, P(U)) defining causal mechanism' },
      { term: 'do(X=x)', definition: 'Intervention that sets X=x, breaking incoming arrows' },
      { term: 'Structural equation', definition: 'X := f(Parents, U) defines how X is generated' }
    ],
    examples: {
      python: `import numpy as np

np.random.seed(42)
n = 3000

# SCM: Z → X → Y, Z → Y (X and Y confounded by Z)

# Structural equations (causal mechanism)
U_Z = np.random.normal(0, 1, n)
U_X = np.random.normal(0, 1, n)
U_Y = np.random.normal(0, 1, n)

Z = U_Z  # Z := f_Z(U_Z)
X = 0.6 * Z + U_X  # X := f_X(Z, U_X)
Y = 1.5 * X + 0.8 * Z + U_Y  # Y := f_Y(X, Z, U_Y)

# Observational: P(Y | X=x) ≠ causal effect
from sklearn.linear_model import LinearRegression
obs_effect = LinearRegression().fit(X.reshape(-1,1), Y).coef_[0]

# Intervention: P(Y | do(X=x)) via backdoor adjustment
# do(X=x) breaks Z → X, so control for Z
causal_effect = LinearRegression().fit(np.c_[X, Z], Y).coef_[0]

print(f"Observational association: {obs_effect:.3f}")
print(f"Causal effect P(Y|do(X)): {causal_effect:.3f}")
print(f"True causal effect: 1.500")
print("\\nBackdoor criterion: Control for Z blocks confounding path")`,
      r: `set.seed(42)
n <- 3000

# Structural equations
Z <- rnorm(n)
X <- 0.6 * Z + rnorm(n)
Y <- 1.5 * X + 0.8 * Z + rnorm(n)

# Observational
obs <- coef(lm(Y ~ X))[2]

# Causal (backdoor adjustment)
causal <- coef(lm(Y ~ X + Z))[2]

cat("Observational:", round(obs, 3), "\\n")
cat("Causal P(Y|do(X)):", round(causal, 3), "\\n")`
    },
    references: [
      { authors: 'Pearl J', title: 'Causality: Models, Reasoning, and Inference', year: 2009, doi: '10.1017/CBO9780511803161' }
    ]
  },

  {
    id: 'framework_principal_stratification',
    title: 'Principal Stratification',
    tier: 'Intermediate',
    description: 'Subgroup effects defined by potential treatment receipt',
    content: `Principal Stratification defines subgroups based on joint potential treatment values.

**Core Idea:**
Classify units by (A^0, A^1) - treatment they would receive under control vs encouragement.

**LATE (Local Average Treatment Effect):**
With binary instrument Z:
- Compliers: A^0=0, A^1=1 (follow encouragement)
- Always-takers: A^0=1, A^1=1
- Never-takers: A^0=0, A^1=0
- Defiers: A^0=1, A^1=0 (ruled out by monotonicity)

**Identification:**
LATE = E[Y^1 - Y^0 | Compliers] identified under:
1. Relevance: E[A | Z=1] > E[A | Z=0]
2. Exclusion: Z affects Y only through A
3. Monotonicity: No defiers
4. Exchangeability of Z

**Compared to ATE:**
- LATE is effect for specific subgroup (compliers)
- ATE is average effect for whole population
- LATE may not generalize beyond compliers

**Applications:**
- Instrumental variables
- Encouragement designs
- Non-compliance in RCTs`,
    prerequisites: ['framework_potential_outcomes'],
    learningObjectives: [
      'Define principal strata',
      'Identify compliers, never-takers, always-takers',
      'Compute LATE with IV assumptions'
    ],
    keyDefinitions: [
      { term: 'Principal Strata', definition: 'Subgroups defined by joint potential treatments' },
      { term: 'Complier', definition: 'Unit with A^0=0, A^1=1' },
      { term: 'LATE', definition: 'Average treatment effect for compliers' }
    ],
    examples: {
      python: `import numpy as np

np.random.seed(42)
n = 5000

# Latent compliance types
compliance = np.random.choice(['complier', 'never', 'always'], n, p=[0.4, 0.4, 0.2])

# Instrument (encouragement)
Z = np.random.binomial(1, 0.5, n)

# Treatment receipt based on compliance type
A = np.where(compliance == 'always', 1,
             np.where(compliance == 'never', 0,
                      Z))  # Compliers follow Z

# Potential outcomes
Y0 = np.random.normal(10, 2, n)
Y1 = Y0 + np.where(compliance == 'complier', 5, 0) + np.random.normal(0, 1, n)
Y_obs = np.where(A == 1, Y1, Y0)

# LATE estimation (Wald estimator)
ITT_Y = Y_obs[Z==1].mean() - Y_obs[Z==0].mean()
ITT_A = A[Z==1].mean() - A[Z==0].mean()
LATE_est = ITT_Y / ITT_A

# True LATE (only for compliers)
true_LATE = (Y1[compliance=='complier'] - Y0[compliance=='complier']).mean()

print(f"True LATE (compliers): {true_LATE:.3f}")
print(f"Estimated LATE: {LATE_est:.3f}")
print(f"\\nCompliance: {(compliance=='complier').mean()*100:.1f}% compliers")`,
      r: `set.seed(42)
n <- 5000

compliance <- sample(c('complier','never','always'), n, prob=c(0.4,0.4,0.2), replace=TRUE)
Z <- rbinom(n, 1, 0.5)
A <- ifelse(compliance=='always', 1, ifelse(compliance=='never', 0, Z))

Y0 <- rnorm(n, 10, 2)
Y1 <- Y0 + ifelse(compliance=='complier', 5, 0) + rnorm(n)
Y_obs <- ifelse(A==1, Y1, Y0)

LATE <- (mean(Y_obs[Z==1]) - mean(Y_obs[Z==0])) / (mean(A[Z==1]) - mean(A[Z==0]))
cat("Estimated LATE:", round(LATE, 3), "\\n")`
    },
    references: [
      { authors: 'Frangakis CE, Rubin DB', title: 'Principal stratification', year: 2002, doi: '10.1111/j.0006-341X.2002.00021.x' }
    ]
  },

  {
    id: 'framework_proximal',
    title: 'Proximal Causal Inference',
    tier: 'Advanced',
    description: 'Identification with unmeasured confounding via proxy variables',
    content: `Proximal inference identifies causal effects when unmeasured confounding exists but proxies are available.

**Setup:**
- U: unmeasured confounder
- Z: treatment confounding proxy (mediates U → A)
- W: outcome confounding proxy (mediates U → Y)

**Key Insight:**
Even without measuring U, if we measure proxies Z and W with certain completeness properties, identification is possible.

**Completeness Conditions:**
1. Treatment confounding bridge: Z captures all U → A confounding
2. Outcome confounding bridge: W captures all U → Y confounding

**Compared to Standard Methods:**
- Standard: Requires measuring all confounders
- Proximal: Can work with proxy variables
- Enables causal inference in settings previously deemed hopeless

**Applications:**
- Electronic health records (proxies for unmeasured health status)
- Negative control designs
- Confounding control when direct measurement impossible`,
    prerequisites: ['framework_scm', 'identification'],
    learningObjectives: [
      'Understand proxy variables for unmeasured confounding',
      'Apply bridge functions for identification',
      'Recognize when proximal methods are needed'
    ],
    keyDefinitions: [
      { term: 'Treatment Proxy Z', definition: 'Variable mediating unmeasured confounder U to treatment A' },
      { term: 'Outcome Proxy W', definition: 'Variable mediating unmeasured confounder U to outcome Y' },
      { term: 'Bridge Function', definition: 'Function capturing confounding pathway through proxies' }
    ],
    examples: {
      python: `import numpy as np

np.random.seed(42)
n = 3000

# Unmeasured confounder U (e.g., frailty)
U = np.random.normal(0, 1, n)

# Proxies
Z = 0.8 * U + np.random.normal(0, 0.5, n)  # Treatment proxy (e.g., prior utilization)
W = 0.7 * U + np.random.normal(0, 0.5, n)  # Outcome proxy (e.g., comorbidity score)

# Treatment and outcome
A = (0.5 * U + 0.3 * Z > 0).astype(int)
Y = 2.0 * A + 0.6 * U + 0.2 * W + np.random.normal(0, 1, n)

# Naive (biased)
from sklearn.linear_model import LinearRegression
naive = LinearRegression().fit(A.reshape(-1,1), Y).coef_[0]

# Proxy adjustment (approximation)
proxy_adj = LinearRegression().fit(np.c_[A, Z, W], Y).coef_[0]

print(f"True causal effect: 2.000")
print(f"Naive (biased by U): {naive:.3f}")
print(f"Proxy-adjusted: {proxy_adj:.3f}")
print("\\nProxies Z and W help recover causal effect despite unmeasured U")`,
      r: `set.seed(42)
n <- 3000

U <- rnorm(n)
Z <- 0.8 * U + rnorm(n, 0, 0.5)
W <- 0.7 * U + rnorm(n, 0, 0.5)

A <- as.integer(0.5*U + 0.3*Z > 0)
Y <- 2*A + 0.6*U + 0.2*W + rnorm(n)

naive <- coef(lm(Y ~ A))[2]
proxy <- coef(lm(Y ~ A + Z + W))[2]

cat("Naive:", round(naive, 3), "\\n")
cat("Proxy-adjusted:", round(proxy, 3), "\\n")`
    },
    references: [
      { authors: 'Miao W et al', title: 'Identifying causal effects with proxy variables', year: 2018, doi: '10.1080/01621459.2017.1401373' },
      { authors: 'Tchetgen Tchetgen EJ et al', title: 'Proximal causal inference', year: 2020, doi: '10.1214/19-STS732' }
    ]
  },

  {
    id: 'framework_bayesian',
    title: 'Bayesian Decision-Theoretic Framework',
    tier: 'Advanced',
    description: 'Causal inference as decision-making under uncertainty',
    content: `Bayesian causal inference integrates prior knowledge, incorporates uncertainty, and optimizes decisions.

**Core Components:**
1. **Prior**: π(θ) on causal parameters
2. **Data**: Likelihood p(Y, A, X | θ)
3. **Posterior**: π(θ | Data) ∝ p(Data | θ) π(θ)
4. **Decision**: Choose action to maximize expected utility

**Advantages:**
- Coherent uncertainty quantification
- Natural integration of external information
- Handles partial identification gracefully
- Optimal decision-making explicit

**Sensitivity Analysis:**
- Prior on degree of unmeasured confounding
- Posterior intervals account for structural uncertainty
- Probabilistic bounds replace point identification

**Value of Information:**
VOI = Expected gain from collecting additional data before deciding

**Compared to Frequentist:**
- Frequentist: Hypothesis tests, confidence intervals
- Bayesian: Probability statements about parameters, decision-optimal actions

**Applications:**
- Clinical trials with adaptive designs
- Policy evaluation with uncertainty
- Optimal treatment regimes`,
    prerequisites: ['framework_potential_outcomes'],
    learningObjectives: [
      'Apply Bayesian inference to causal estimands',
      'Conduct sensitivity analysis with priors',
      'Compute value of information for decisions'
    ],
    keyDefinitions: [
      { term: 'Prior', definition: 'Probability distribution on parameters before seeing data' },
      { term: 'Posterior', definition: 'Updated distribution after observing data' },
      { term: 'Value of Information', definition: 'Expected benefit of obtaining additional data' }
    ],
    examples: {
      python: `import numpy as np

np.random.seed(42)
n = 500

# Data from RCT
A = np.random.binomial(1, 0.5, n)
Y = 2.5 * A + np.random.normal(0, 2, n)

# Bayesian inference on ATE
# Prior: ATE ~ N(0, 10)  (weakly informative)
prior_mean, prior_var = 0, 10

# Likelihood for ATE (simplified)
Y1, Y0 = Y[A==1], Y[A==0]
data_est = Y1.mean() - Y0.mean()
data_var = Y1.var()/len(Y1) + Y0.var()/len(Y0)

# Posterior: Normal conjugate update
post_var = 1 / (1/prior_var + 1/data_var)
post_mean = post_var * (prior_mean/prior_var + data_est/data_var)

# Frequentist (for comparison)
freq_est = data_est

print(f"Prior: N({prior_mean}, {np.sqrt(prior_var):.2f})")
print(f"Data estimate: {data_est:.3f}")
print(f"\\nPosterior: N({post_mean:.3f}, {np.sqrt(post_var):.3f})")
print(f"Frequentist point estimate: {freq_est:.3f}")
print(f"\\n95% Bayesian Credible Interval: [{post_mean - 1.96*np.sqrt(post_var):.2f}, {post_mean + 1.96*np.sqrt(post_var):.2f}]")`,
      r: `set.seed(42)
n <- 500

A <- rbinom(n, 1, 0.5)
Y <- 2.5*A + rnorm(n, 0, 2)

# Prior
prior_mean <- 0
prior_var <- 10

# Data
Y1 <- Y[A==1]
Y0 <- Y[A==0]
data_est <- mean(Y1) - mean(Y0)
data_var <- var(Y1)/length(Y1) + var(Y0)/length(Y0)

# Posterior
post_var <- 1 / (1/prior_var + 1/data_var)
post_mean <- post_var * (prior_mean/prior_var + data_est/data_var)

cat("Posterior mean:", round(post_mean, 3), "\\n")
cat("Posterior SD:", round(sqrt(post_var), 3), "\\n")`
    },
    references: [
      { authors: 'Gustafson P', title: 'Bayesian Inference for Partially Identified Models', year: 2015, doi: '10.1201/b18308' }
    ]
  }
];
