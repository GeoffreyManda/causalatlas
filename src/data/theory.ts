export interface TheoryTopic {
  id: string;
  title: string;
  tier: 'Foundational' | 'Intermediate' | 'Advanced' | 'Frontier';
  description: string;
  content: string;
  prerequisites: string[];
  learningObjectives: string[];
  keyDefinitions: { term: string; definition: string }[];
  
  // New comprehensive sections (to be populated)
  backgroundMotivation?: string;
  historicalContext?: string;
  conditionsAssumptions?: string;
  dataStructureDesign?: string;
  targetParameter?: string;
  identificationStrategy?: string;
  estimationPlan?: string;
  diagnosticsValidation?: string;
  sensitivityRobustness?: string;
  ethicsGovernance?: string;
  
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
  },

  // ========== PROPENSITY SCORE METHODS ==========
  {
    id: 'unconfoundedness_propensity_score',
    title: 'Unconfoundedness and the Propensity Score',
    tier: 'Intermediate',
    description: 'The ignorability assumption and propensity score methods for causal inference',
    content: `## The Unconfoundedness Assumption

**Unconfoundedness** (also called **ignorability** or **conditional exchangeability**) is the fundamental assumption that enables causal inference from observational data.

**Formal Definition:**
(Y¹, Y⁰) ⊥ A | X

This states: conditional on measured covariates X, treatment assignment A is independent of potential outcomes (Y¹, Y⁰).

**Intuition:** Once we control for X, treatment is "as good as random" - there are no unmeasured confounders affecting both treatment and outcome.

---

## The Propensity Score

The **propensity score** is the probability of receiving treatment given covariates:

e(X) = P(A=1 | X)

**Rosenbaum & Rubin's (1983) Key Result:**
If unconfoundedness holds given X, then it also holds given e(X):

(Y¹, Y⁰) ⊥ A | X  ⟹  (Y¹, Y⁰) ⊥ A | e(X)

**Why This Matters:** Instead of conditioning on high-dimensional X, we can condition on the scalar e(X). This is a powerful **dimension reduction**.

---

## Propensity Score Methods

### 1. **Matching**
Pair treated and control units with similar propensity scores.
- **Nearest neighbor:** Match each treated unit to closest control
- **Caliper matching:** Only match within specified distance
- **Optimal matching:** Minimize total distance

**Advantages:** Intuitive, transparent
**Limitations:** Discards unmatched units, potential bias if poor matches

### 2. **Stratification/Subclassification**
Divide units into strata based on propensity score quintiles (or finer), estimate effects within strata, then aggregate.

**Rosenbaum & Rubin (1984):** 5 strata remove ~90% of bias due to confounding.

### 3. **Inverse Probability Weighting (IPW)**
Weight each unit by inverse of probability of their observed treatment:

w_i = A_i / e(X_i) + (1 - A_i) / (1 - e(X_i))

This creates a **pseudo-population** where treatment is independent of X.

**ATE Estimator:**
τ̂_IPW = (1/n) Σ [A_i Y_i / e(X_i) - (1-A_i) Y_i / (1-e(X_i))]

**Advantages:** Uses all data, targets population ATE
**Limitations:** Sensitive to extreme weights (positivity violations)

### 4. **Covariate Adjustment / Regression**
Include propensity score as covariate in outcome regression:

E[Y | A, e(X)]

Can be combined with other adjustment methods.

### 5. **Doubly Robust Methods**
Combine propensity score with outcome regression (e.g., AIPW, TMLE). Correct if **either** model is correctly specified.

---

## Key Assumptions

**1. Unconfoundedness:** (Y¹, Y⁰) ⊥ A | X
- All confounders are measured and included in X
- Requires domain knowledge and causal reasoning (DAGs)

**2. Positivity/Overlap:** 0 < e(X) < 1 for all X
- Every covariate pattern has non-zero probability of both treatments
- Violations: extreme weights, poor matches, limited generalizability

**3. SUTVA:** No interference, well-defined treatment

---

## Diagnostics

### Balance Checks
After propensity score adjustment, check if covariates are balanced:
- **Standardized mean differences:** Should be < 0.1
- **Variance ratios:** Should be close to 1
- **Overlap plots:** Visualize propensity score distributions

### Sensitivity Analysis
- **Rosenbaum bounds:** How strong must unmeasured confounding be to change conclusions?
- **E-values:** Minimum strength of unmeasured confounder to explain away effect
- **Simulation-based:** Posit unmeasured confounder and assess impact

---

## Practical Tips

1. **Propensity score estimation:**
   - Use flexible models (boosting, super learner)
   - Include interactions and non-linearities
   - Maximize covariate balance, not prediction accuracy

2. **Trimming:** Discard units with extreme propensity scores (e.g., < 0.1 or > 0.9) to improve overlap

3. **Weight stabilization:** Use stabilized weights to reduce variance:
   w_i = P(A_i) / P(A_i | X_i)

4. **Combine methods:** Use propensity scores for initial balance, then outcome regression for final estimate

---

## When Unconfoundedness Fails

If unmeasured confounding exists:
- **Instrumental variables:** Use natural experiments
- **Difference-in-differences:** Control for time-invariant confounding
- **Regression discontinuity:** Exploit threshold-based assignment
- **Sensitivity analysis:** Quantify robustness to violations
- **Proximal causal inference:** Use proxy variables`,
    prerequisites: ['intro_causal_inference', 'framework_potential_outcomes'],
    learningObjectives: [
      'Define unconfoundedness and understand its role in causal inference',
      'Explain the propensity score and its dimension reduction property',
      'Apply propensity score methods: matching, IPW, stratification',
      'Assess balance and overlap using diagnostics',
      'Conduct sensitivity analysis for unmeasured confounding'
    ],
    keyDefinitions: [
      { term: 'Unconfoundedness', definition: '(Y¹, Y⁰) ⊥ A | X - treatment assignment is independent of potential outcomes given measured covariates' },
      { term: 'Propensity Score', definition: 'e(X) = P(A=1 | X) - probability of treatment given covariates' },
      { term: 'Positivity', definition: '0 < e(X) < 1 - all covariate patterns have positive probability of both treatments' },
      { term: 'Balancing Score', definition: 'A function of covariates such that conditioning on it balances treatment groups' },
      { term: 'Inverse Probability Weighting', definition: 'Weighting observations by 1/e(X) or 1/(1-e(X)) to create pseudo-population' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
n = 2000

# ========== SIMULATE DATA WITH CONFOUNDING ==========
# Confounders
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
X = np.column_stack([X1, X2])

# Treatment depends on X (selection bias)
logit_A = 0.5 + 0.8*X1 - 0.6*X2
prob_A = 1 / (1 + np.exp(-logit_A))
A = np.random.binomial(1, prob_A)

# Outcome depends on X AND A
# True ATE = 2.0
Y = 5.0 + 2.0*A + 1.5*X1 + 1.2*X2 + np.random.normal(0, 1, n)

print("=== CONFOUNDED DATA ===")
print(f"True ATE: 2.0")

# Naive estimate (biased!)
naive = Y[A==1].mean() - Y[A==0].mean()
print(f"Naive difference: {naive:.3f} (BIASED)")

# ========== PROPENSITY SCORE ESTIMATION ==========
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, A)
propensity_scores = ps_model.predict_proba(X)[:, 1]

print(f"\\n=== PROPENSITY SCORES ===")
print(f"Mean PS (treated): {propensity_scores[A==1].mean():.3f}")
print(f"Mean PS (control): {propensity_scores[A==0].mean():.3f}")
print(f"PS range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")

# ========== METHOD 1: IPW ==========
weights = A / propensity_scores + (1-A) / (1-propensity_scores)
# Trim extreme weights
weights = np.clip(weights, 0, 10)

ate_ipw = np.average(Y[A==1], weights=weights[A==1]) - \\
          np.average(Y[A==0], weights=weights[A==0])
print(f"\\n=== IPW ESTIMATE ===")
print(f"ATE (IPW): {ate_ipw:.3f}")

# ========== METHOD 2: MATCHING ==========
# 1:1 nearest neighbor matching on propensity score
ps_treated = propensity_scores[A==1].reshape(-1, 1)
ps_control = propensity_scores[A==0].reshape(-1, 1)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(ps_control)
distances, indices = nn.kneighbors(ps_treated)

Y_treated_matched = Y[A==1]
Y_control_matched = Y[A==0][indices.flatten()]

ate_matching = (Y_treated_matched - Y_control_matched).mean()
print(f"\\n=== MATCHING ESTIMATE ===")
print(f"ATE (1:1 Matching): {ate_matching:.3f}")

# ========== METHOD 3: STRATIFICATION ==========
# Divide into 5 strata by PS quintiles
strata = np.digitize(propensity_scores, 
                     np.percentile(propensity_scores, [20, 40, 60, 80]))

ate_strat = 0
for s in range(5):
    mask = strata == s
    if A[mask].sum() > 0 and (1-A[mask]).sum() > 0:
        effect_s = Y[mask & (A==1)].mean() - Y[mask & (A==0)].mean()
        ate_strat += effect_s * mask.sum() / n

print(f"\\n=== STRATIFICATION ESTIMATE ===")
print(f"ATE (5 Strata): {ate_strat:.3f}")

# ========== BALANCE CHECK ==========
# Standardized mean difference before/after
def smd(x, a):
    m1, m0 = x[a==1].mean(), x[a==0].mean()
    s1, s0 = x[a==1].std(), x[a==0].std()
    return (m1 - m0) / np.sqrt((s1**2 + s0**2) / 2)

print(f"\\n=== BALANCE (Std Mean Diff) ===")
print(f"Before adjustment: X1={smd(X1, A):.3f}, X2={smd(X2, A):.3f}")

# After IPW
smd_x1_ipw = smd(X1, A) * np.sqrt(weights.mean())  # Approximation
smd_x2_ipw = smd(X2, A) * np.sqrt(weights.mean())
print(f"After IPW: ~balanced (weights applied)")`,
      r: `library(MatchIt)
library(WeightIt)

set.seed(42)
n <- 2000

# ========== SIMULATE DATA ==========
X1 <- rnorm(n)
X2 <- rnorm(n)

# Treatment with confounding
logit_A <- 0.5 + 0.8*X1 - 0.6*X2
prob_A <- 1 / (1 + exp(-logit_A))
A <- rbinom(n, 1, prob_A)

# Outcome (True ATE = 2.0)
Y <- 5.0 + 2.0*A + 1.5*X1 + 1.2*X2 + rnorm(n)

df <- data.frame(Y, A, X1, X2)

cat("=== TRUE ATE: 2.0 ===\\n")
cat("Naive estimate:", round(mean(Y[A==1]) - mean(Y[A==0]), 3), "\\n\\n")

# ========== PROPENSITY SCORE ==========
ps_model <- glm(A ~ X1 + X2, family = binomial(), data = df)
df$ps <- predict(ps_model, type = "response")

cat("=== PROPENSITY SCORES ===\\n")
cat("Mean PS (treated):", round(mean(df$ps[A==1]), 3), "\\n")
cat("Mean PS (control):", round(mean(df$ps[A==0]), 3), "\\n\\n")

# ========== METHOD 1: IPW ==========
df$weights <- ifelse(A == 1, 1/df$ps, 1/(1-df$ps))
df$weights <- pmin(df$weights, 10)  # Trim

ate_ipw <- weighted.mean(Y[A==1], df$weights[A==1]) - 
           weighted.mean(Y[A==0], df$weights[A==0])
cat("=== IPW ===\\n")
cat("ATE (IPW):", round(ate_ipw, 3), "\\n\\n")

# ========== METHOD 2: MATCHING ==========
m_out <- matchit(A ~ X1 + X2, data = df, method = "nearest", distance = "logit")
matched_data <- match.data(m_out)

ate_matching <- mean(matched_data$Y[matched_data$A==1]) - 
                mean(matched_data$Y[matched_data$A==0])
cat("=== MATCHING ===\\n")
cat("ATE (Matching):", round(ate_matching, 3), "\\n\\n")

# ========== BALANCE CHECK ==========
# SMD function
smd <- function(x, a) {
  m1 <- mean(x[a==1])
  m0 <- mean(x[a==0])
  s1 <- sd(x[a==1])
  s0 <- sd(x[a==0])
  (m1 - m0) / sqrt((s1^2 + s0^2) / 2)
}

cat("=== BALANCE (Std Mean Diff) ===\\n")
cat("Before: X1=", round(smd(X1, A), 3), ", X2=", round(smd(X2, A), 3), "\\n")
cat("After matching: improved balance\\n")`
    },
    references: [
      { authors: 'Rosenbaum PR, Rubin DB', title: 'The central role of the propensity score in observational studies for causal effects', year: 1983, doi: '10.1093/biomet/70.1.41' },
      { authors: 'Rosenbaum PR, Rubin DB', title: 'Reducing bias in observational studies using subclassification on the propensity score', year: 1984, doi: '10.1080/01621459.1984.10478078' },
      { authors: 'Austin PC', title: 'An Introduction to Propensity Score Methods for Reducing Confounding', year: 2011, doi: '10.1080/00273171.2011.568786' },
      { authors: 'Stuart EA', title: 'Matching methods for causal inference: A review', year: 2010, doi: '10.1214/09-STS313' }
    ]
  },

  // ========== FOUNDATIONAL TOPICS (Hernán & Robins Part I) ==========
  {
    id: 'randomized_experiments',
    title: 'Randomized Experiments',
    tier: 'Foundational',
    description: 'The gold standard for causal inference - design, analysis, and interpretation of RCTs',
    content: `*Content pending. This topic will cover: randomization principles, conditional randomization, standardization, and inverse probability weighting in experimental settings.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Understand why randomization enables causal inference',
      'Distinguish unconditional and conditional randomization',
      'Apply standardization and IPW in randomized experiments',
      'Interpret intention-to-treat and per-protocol effects'
    ],
    keyDefinitions: [
      { term: 'Randomization', definition: 'Random assignment of treatment to ensure exchangeability' },
      { term: 'Conditional randomization', definition: 'Randomization within strata defined by covariates' },
      { term: 'ITT effect', definition: 'Effect of treatment assignment regardless of compliance' }
    ],
    examples: {
      python: `# Placeholder - RCT analysis example`,
      r: `# Placeholder - RCT analysis example`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 2', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'observational_studies',
    title: 'Observational Studies',
    tier: 'Foundational',
    description: 'Identifiability conditions for causal inference from observational data',
    content: `*Content pending. This topic will cover: exchangeability, positivity, consistency, SUTVA, and the target trial framework.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'State the three core identifiability conditions',
      'Understand exchangeability and its violations',
      'Assess positivity violations and their implications',
      'Apply the target trial framework'
    ],
    keyDefinitions: [
      { term: 'Exchangeability', definition: '(Y¹,Y⁰) ⊥ A | X - no unmeasured confounding' },
      { term: 'Positivity', definition: 'P(A=a|X=x) > 0 for all a,x with P(X=x) > 0' },
      { term: 'Consistency', definition: 'If A=a, then Y=Yᵃ' }
    ],
    examples: {
      python: `# Placeholder - observational study analysis`,
      r: `# Placeholder - observational study analysis`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 3', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'effect_modification',
    title: 'Effect Modification and Heterogeneity',
    tier: 'Intermediate',
    description: 'Heterogeneous treatment effects across subgroups and effect modification',
    content: `*Content pending. This topic will cover: stratification, matching, CATE estimation, heterogeneous treatment effects, and meta-learners.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Define effect modification vs interaction',
      'Identify effect modification via stratification',
      'Estimate conditional average treatment effects (CATE)',
      'Apply S-learner, T-learner, X-learner, R-learner'
    ],
    keyDefinitions: [
      { term: 'Effect modification', definition: 'Treatment effect varies across levels of covariate V' },
      { term: 'CATE', definition: 'E[Y¹-Y⁰|X=x] - conditional average treatment effect' },
      { term: 'HTE', definition: 'Heterogeneous treatment effects - variation in individual effects' }
    ],
    examples: {
      python: `# Placeholder - CATE estimation with meta-learners`,
      r: `# Placeholder - CATE estimation`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 4', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' },
      { authors: 'Künzel SR et al', title: 'Metalearners for estimating heterogeneous treatment effects', year: 2019, doi: '10.1073/pnas.1804597116' }
    ]
  },

  {
    id: 'interaction',
    title: 'Interaction and Sufficient Causes',
    tier: 'Intermediate',
    description: 'Causal interaction, sufficient causes, and mechanistic interaction',
    content: `*Content pending. This topic will cover: joint interventions, sufficient cause interaction, counterfactual response types, and mechanistic frameworks.*`,
    prerequisites: ['effect_modification'],
    learningObjectives: [
      'Distinguish effect modification from interaction',
      'Understand sufficient cause framework',
      'Identify synergism and antagonism',
      'Apply counterfactual response type analysis'
    ],
    keyDefinitions: [
      { term: 'Interaction', definition: 'Effect of joint intervention differs from sum of individual effects' },
      { term: 'Sufficient cause', definition: 'Minimal set of conditions that inevitably produce outcome' },
      { term: 'Synergism', definition: 'Positive interaction - combined effect exceeds additive' }
    ],
    examples: {
      python: `# Placeholder - interaction analysis`,
      r: `# Placeholder - interaction analysis`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 5', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'confounding',
    title: 'Confounding and the Backdoor Criterion',
    tier: 'Foundational',
    description: 'Structure of confounding, backdoor criterion, and confounding adjustment',
    content: `*Content pending. This topic will cover: confounding structure, backdoor criterion, single-world intervention graphs (SWIGs), and adjustment strategies.*`,
    prerequisites: ['dags_basics'],
    learningObjectives: [
      'Identify confounding in DAGs',
      'Apply backdoor criterion',
      'Understand SWIGs',
      'Choose valid adjustment sets'
    ],
    keyDefinitions: [
      { term: 'Confounding', definition: 'Common cause of treatment and outcome creating spurious association' },
      { term: 'Backdoor path', definition: 'Non-causal path from treatment to outcome' },
      { term: 'Backdoor criterion', definition: 'Set Z blocks all backdoor paths and contains no descendants of A' }
    ],
    examples: {
      python: `# Placeholder - confounding adjustment`,
      r: `# Placeholder - confounding adjustment`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 7', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'selection_bias',
    title: 'Selection Bias and Collider Stratification',
    tier: 'Intermediate',
    description: 'Selection bias, censoring, collider bias, and adjustment strategies',
    content: `*Content pending. This topic will cover: selection bias structure, collider stratification bias, censoring mechanisms, and inverse probability of censoring weighting.*`,
    prerequisites: ['dags_basics', 'confounding'],
    learningObjectives: [
      'Identify selection bias in DAGs',
      'Understand collider stratification bias',
      'Adjust for informative censoring',
      'Apply inverse probability of selection weighting'
    ],
    keyDefinitions: [
      { term: 'Selection bias', definition: 'Bias from conditioning on common effect of treatment and outcome' },
      { term: 'Collider', definition: 'Variable caused by two other variables' },
      { term: 'Collider stratification bias', definition: 'Spurious association created by conditioning on collider' }
    ],
    examples: {
      python: `# Placeholder - selection bias correction`,
      r: `# Placeholder - selection bias correction`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 8', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'measurement_bias',
    title: 'Measurement Bias and Misclassification',
    tier: 'Advanced',
    description: 'Measurement error, misclassification, and their impact on causal inference',
    content: `*Content pending. This topic will cover: measurement error structures, mismeasured confounders, noncausal diagrams, and correction methods.*`,
    prerequisites: ['confounding', 'selection_bias'],
    learningObjectives: [
      'Understand measurement error impact on estimates',
      'Identify mismeasured confounders in DAGs',
      'Distinguish causal vs noncausal arrows',
      'Apply measurement error correction techniques'
    ],
    keyDefinitions: [
      { term: 'Measurement error', definition: 'Discrepancy between observed and true variable value' },
      { term: 'Noncausal arrow', definition: 'Arrow representing measurement process, not causation' },
      { term: 'Classical measurement error', definition: 'Random error independent of true value' }
    ],
    examples: {
      python: `# Placeholder - measurement error correction`,
      r: `# Placeholder - measurement error correction`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 9', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'random_variability',
    title: 'Random Variability and Finite Sample Inference',
    tier: 'Intermediate',
    description: 'Identification vs estimation, sampling variability, and statistical inference',
    content: `*Content pending. This topic will cover: identification vs estimation distinction, finite sample inference, conditionality principle, curse of dimensionality.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Distinguish identification from estimation',
      'Quantify random variability in estimates',
      'Apply conditionality principle',
      'Understand curse of dimensionality'
    ],
    keyDefinitions: [
      { term: 'Identification', definition: 'Expressing causal estimand as function of observable distribution' },
      { term: 'Estimation', definition: 'Computing numerical value from finite sample' },
      { term: 'Sampling variability', definition: 'Variation in estimates across repeated samples' }
    ],
    examples: {
      python: `# Placeholder - bootstrap inference`,
      r: `# Placeholder - bootstrap inference`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 10', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  // ========== PARAMETRIC & SEMIPARAMETRIC METHODS ==========
  {
    id: 'why_model',
    title: 'Why Model? Parametric vs Nonparametric Estimation',
    tier: 'Intermediate',
    description: 'Role of models, bias-variance tradeoff, smoothing, and model selection',
    content: `*Content pending. This topic will cover: parametric vs nonparametric estimation, conditional mean estimation, smoothing techniques, bias-variance tradeoff.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Understand role of statistical models',
      'Compare parametric and nonparametric approaches',
      'Apply bias-variance tradeoff',
      'Choose appropriate model complexity'
    ],
    keyDefinitions: [
      { term: 'Parametric model', definition: 'Model with finite-dimensional parameter' },
      { term: 'Nonparametric', definition: 'Model allowing infinite-dimensional parameter space' },
      { term: 'Bias-variance tradeoff', definition: 'Balance between model bias and estimation variance' }
    ],
    examples: {
      python: `# Placeholder - model comparison`,
      r: `# Placeholder - model comparison`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 11', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'marginal_structural_models',
    title: 'IP Weighting and Marginal Structural Models',
    tier: 'Intermediate',
    description: 'Inverse probability weighting, marginal structural models, and stabilized weights',
    content: `*Content pending. This topic will cover: IP weight estimation, stabilized weights, marginal structural models, effect modification in MSMs.*`,
    prerequisites: ['unconfoundedness_propensity_score'],
    learningObjectives: [
      'Estimate inverse probability weights',
      'Fit marginal structural models',
      'Apply stabilized weights',
      'Model effect modification in MSMs'
    ],
    keyDefinitions: [
      { term: 'Marginal structural model', definition: 'Model for counterfactual outcome means marginalized over confounders' },
      { term: 'Stabilized weights', definition: 'Weights with numerator to reduce variance' },
      { term: 'IP weighting', definition: 'Weighting by inverse of treatment probability' }
    ],
    examples: {
      python: `# Placeholder - MSM estimation`,
      r: `# Placeholder - MSM estimation`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 12', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'parametric_g_formula',
    title: 'Standardization and the Parametric G-Formula',
    tier: 'Intermediate',
    description: 'G-formula for time-fixed treatments, standardization, and outcome modeling',
    content: `*Content pending. This topic will cover: standardization method, parametric g-formula, outcome regression modeling, comparison with IPW.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Apply standardization to estimate causal effects',
      'Implement parametric g-formula',
      'Compare g-formula with IP weighting',
      'Model outcome conditional distributions'
    ],
    keyDefinitions: [
      { term: 'Standardization', definition: 'Averaging covariate-conditional effects over covariate distribution' },
      { term: 'G-formula', definition: 'Formula expressing causal effect via standardization' },
      { term: 'Outcome model', definition: 'Model for E[Y|A,X]' }
    ],
    examples: {
      python: `# Placeholder - g-formula implementation`,
      r: `# Placeholder - g-formula implementation`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 13', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'structural_nested_models',
    title: 'G-Estimation of Structural Nested Models',
    tier: 'Advanced',
    description: 'Structural nested models, g-estimation, and rank preservation',
    content: `*Content pending. This topic will cover: structural nested mean models, g-estimation procedure, rank preservation, two-stage estimation.*`,
    prerequisites: ['parametric_g_formula'],
    learningObjectives: [
      'Specify structural nested models',
      'Apply g-estimation algorithm',
      'Understand rank preservation',
      'Implement multi-parameter g-estimation'
    ],
    keyDefinitions: [
      { term: 'Structural nested model', definition: 'Model for individual treatment effect conditional on covariates' },
      { term: 'G-estimation', definition: 'Estimation method using orthogonality conditions' },
      { term: 'Rank preservation', definition: 'Assumption that treatment does not change outcome ranking' }
    ],
    examples: {
      python: `# Placeholder - g-estimation`,
      r: `# Placeholder - g-estimation`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 14', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'augmented_ipw',
    title: 'Augmented IPW and Doubly Robust Estimation',
    tier: 'Advanced',
    description: 'AIPW estimators, doubly robust property, and semiparametric efficiency',
    content: `*Content pending. This topic will cover: augmented inverse probability weighting, doubly robust estimation, efficient influence function, one-step and TMLE estimators.*`,
    prerequisites: ['marginal_structural_models', 'parametric_g_formula'],
    learningObjectives: [
      'Understand doubly robust property',
      'Implement AIPW estimators',
      'Apply targeted maximum likelihood estimation (TMLE)',
      'Achieve semiparametric efficiency'
    ],
    keyDefinitions: [
      { term: 'Doubly robust', definition: 'Consistent if either propensity or outcome model correct' },
      { term: 'AIPW', definition: 'Augmented inverse probability weighted estimator' },
      { term: 'Efficient influence function', definition: 'Canonical gradient determining efficiency bound' }
    ],
    examples: {
      python: `# Placeholder - AIPW/TMLE implementation`,
      r: `# Placeholder - AIPW/TMLE implementation`
    },
    references: [
      { authors: 'Robins JM et al', title: 'Doubly robust estimation', year: 2007, doi: '10.1198/073500106000000533' },
      { authors: 'van der Laan MJ, Rose S', title: 'Targeted Learning', year: 2011, doi: '10.1007/978-1-4419-9782-1' }
    ]
  },

  {
    id: 'instrumental_variables',
    title: 'Instrumental Variables and LATE',
    tier: 'Advanced',
    description: 'IV estimation, local average treatment effects, and complier analysis',
    content: `*Content pending. This topic will cover: three IV conditions, Wald estimator, LATE identification, monotonicity, principal stratification for IV.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'State three instrumental conditions',
      'Estimate effects with instrumental variables',
      'Interpret local average treatment effects',
      'Apply complier average causal effect framework'
    ],
    keyDefinitions: [
      { term: 'Instrument', definition: 'Variable affecting treatment but not outcome except through treatment' },
      { term: 'LATE', definition: 'Local average treatment effect for compliers' },
      { term: 'Complier', definition: 'Unit whose treatment responds to instrument' }
    ],
    examples: {
      python: `# Placeholder - IV estimation`,
      r: `# Placeholder - IV estimation`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 16', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' },
      { authors: 'Angrist JD et al', title: 'Identification of causal effects using instrumental variables', year: 1996, doi: '10.1080/01621459.1996.10476902' }
    ]
  },

  {
    id: 'survival_analysis',
    title: 'Causal Survival Analysis',
    tier: 'Advanced',
    description: 'Survival outcomes, censoring, hazards, and time-to-event causal effects',
    content: `*Content pending. This topic will cover: hazards vs risks, censoring adjustment, survival IP weighting, g-formula for survival, g-estimation with censoring.*`,
    prerequisites: ['marginal_structural_models'],
    learningObjectives: [
      'Define causal effects for survival outcomes',
      'Distinguish hazards and risks',
      'Adjust for informative censoring',
      'Apply survival-specific g-methods'
    ],
    keyDefinitions: [
      { term: 'Hazard', definition: 'Instantaneous failure rate conditional on survival to time t' },
      { term: 'Risk', definition: 'Cumulative probability of failure by time t' },
      { term: 'Censoring', definition: 'Loss to follow-up before outcome occurs' }
    ],
    examples: {
      python: `# Placeholder - causal survival analysis`,
      r: `# Placeholder - causal survival analysis`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 17', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'variable_selection',
    title: 'Variable Selection and High-Dimensional Data',
    tier: 'Advanced',
    description: 'Covariate selection, machine learning in causal inference, double/debiased ML',
    content: `*Content pending. This topic will cover: goals of variable selection, bias-inducing variables, causal ML integration, doubly robust ML estimators.*`,
    prerequisites: ['augmented_ipw'],
    learningObjectives: [
      'Identify variables that induce or amplify bias',
      'Integrate machine learning in causal inference',
      'Apply double/debiased machine learning',
      'Navigate high-dimensional confounder spaces'
    ],
    keyDefinitions: [
      { term: 'Double ML', definition: 'ML-based estimation with Neyman orthogonality and cross-fitting' },
      { term: 'Bias amplification', definition: 'Adjusting for certain variables increases bias' },
      { term: 'Cross-fitting', definition: 'Sample splitting to prevent overfitting bias' }
    ],
    examples: {
      python: `# Placeholder - double ML`,
      r: `# Placeholder - double ML`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 18', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' },
      { authors: 'Chernozhukov V et al', title: 'Double/debiased machine learning', year: 2018, doi: '10.1111/ectj.12097' }
    ]
  },

  // ========== TIME-VARYING TREATMENTS ==========
  {
    id: 'time_varying_treatments',
    title: 'Time-Varying Treatments',
    tier: 'Advanced',
    description: 'Treatment strategies, sequential exchangeability, and time-varying confounding',
    content: `*Content pending. This topic will cover: treatment strategies/regimes, sequentially randomized experiments, sequential exchangeability, time-varying confounding.*`,
    prerequisites: ['marginal_structural_models'],
    learningObjectives: [
      'Define causal effects of treatment strategies',
      'Understand sequential exchangeability',
      'Identify time-varying confounding',
      'Specify dynamic treatment regimes'
    ],
    keyDefinitions: [
      { term: 'Treatment strategy', definition: 'Rule specifying treatment at each time based on history' },
      { term: 'Sequential exchangeability', definition: 'Exchangeability at each time conditional on past' },
      { term: 'Time-varying confounder', definition: 'Confounder affected by past treatment' }
    ],
    examples: {
      python: `# Placeholder - time-varying treatment analysis`,
      r: `# Placeholder - time-varying treatment analysis`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 19', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'treatment_confounder_feedback',
    title: 'Treatment-Confounder Feedback',
    tier: 'Advanced',
    description: 'Time-dependent confounding affected by prior treatment and traditional methods failure',
    content: `*Content pending. This topic will cover: treatment-confounder feedback, why traditional methods fail, necessity of g-methods.*`,
    prerequisites: ['time_varying_treatments'],
    learningObjectives: [
      'Identify treatment-confounder feedback',
      'Understand why traditional adjustment fails',
      'Recognize when g-methods are needed',
      'Apply appropriate longitudinal methods'
    ],
    keyDefinitions: [
      { term: 'Treatment-confounder feedback', definition: 'Past treatment affects future confounders' },
      { term: 'Time-dependent confounding', definition: 'Confounder at time t affected by treatment before t' },
      { term: 'G-methods', definition: 'Methods handling time-dependent confounding' }
    ],
    examples: {
      python: `# Placeholder - feedback demonstration`,
      r: `# Placeholder - feedback demonstration`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 20', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'g_methods_longitudinal',
    title: 'G-Methods for Time-Varying Treatments',
    tier: 'Frontier',
    description: 'G-formula, IP weighting, and g-estimation for longitudinal data',
    content: `*Content pending. This topic will cover: longitudinal g-formula, time-varying IPW, doubly robust methods for longitudinal data, g-estimation for time-varying treatments.*`,
    prerequisites: ['treatment_confounder_feedback'],
    learningObjectives: [
      'Apply g-formula to time-varying treatments',
      'Implement time-varying IP weighting',
      'Use doubly robust estimators longitudinally',
      'Apply g-estimation in longitudinal settings'
    ],
    keyDefinitions: [
      { term: 'Longitudinal g-formula', definition: 'Iterative application of g-formula over time' },
      { term: 'Time-varying IPW', definition: 'Product of time-specific inverse probability weights' },
      { term: 'Sequential doubly robust', definition: 'Doubly robust at each time point' }
    ],
    examples: {
      python: `# Placeholder - g-methods longitudinal`,
      r: `# Placeholder - g-methods longitudinal`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 21', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'target_trial_emulation',
    title: 'Target Trial Emulation',
    tier: 'Advanced',
    description: 'Designing observational studies to emulate randomized trials',
    content: `*Content pending. This topic will cover: target trial framework, protocol specification, time-zero alignment, sustained vs point treatment strategies.*`,
    prerequisites: ['time_varying_treatments'],
    learningObjectives: [
      'Specify target trial protocol',
      'Emulate trials with observational data',
      'Avoid immortal time bias',
      'Design time-zero properly'
    ],
    keyDefinitions: [
      { term: 'Target trial', definition: 'Hypothetical RCT the observational study aims to emulate' },
      { term: 'Time zero', definition: 'When treatment strategy eligibility begins' },
      { term: 'Immortal time bias', definition: 'Bias from survival requirement in treatment group' }
    ],
    examples: {
      python: `# Placeholder - target trial emulation`,
      r: `# Placeholder - target trial emulation`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 22', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  {
    id: 'causal_mediation',
    title: 'Causal Mediation Analysis',
    tier: 'Advanced',
    description: 'Direct and indirect effects, mediation analysis, and interventionist approach',
    content: `*Content pending. This topic will cover: natural direct/indirect effects, controlled direct effects, sequential ignorability, interventionist mediation theory.*`,
    prerequisites: ['parametric_g_formula'],
    learningObjectives: [
      'Define direct and indirect effects',
      'Understand sequential ignorability',
      'Implement mediation estimators',
      'Interpret mediation analysis results'
    ],
    keyDefinitions: [
      { term: 'Natural direct effect', definition: 'Effect of treatment not mediated through M' },
      { term: 'Natural indirect effect', definition: 'Effect of treatment operating through M' },
      { term: 'Sequential ignorability', definition: 'Ignorability for treatment and mediator conditional on covariates' }
    ],
    examples: {
      python: `# Placeholder - mediation analysis`,
      r: `# Placeholder - mediation analysis`
    },
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If - Chapter 23', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
    ]
  },

  // ========== ADDITIONAL ADVANCED TOPICS ==========
  {
    id: 'regression_discontinuity',
    title: 'Regression Discontinuity Designs',
    tier: 'Advanced',
    description: 'Sharp and fuzzy RDD, identification at cutoffs, bandwidth selection',
    content: `*Content pending. This topic will cover: sharp vs fuzzy RDD, local randomization, continuity assumptions, bandwidth selection, inference.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Identify RDD opportunities',
      'Distinguish sharp and fuzzy designs',
      'Apply local polynomial estimation',
      'Conduct RDD diagnostics and sensitivity'
    ],
    keyDefinitions: [
      { term: 'Sharp RDD', definition: 'Treatment deterministically assigned at cutoff' },
      { term: 'Fuzzy RDD', definition: 'Treatment probability changes discontinuously at cutoff' },
      { term: 'Bandwidth', definition: 'Window around cutoff used for estimation' }
    ],
    examples: {
      python: `# Placeholder - RDD analysis`,
      r: `# Placeholder - RDD analysis`
    },
    references: [
      { authors: 'Lee DS, Lemieux T', title: 'Regression discontinuity designs in economics', year: 2010, doi: '10.1257/jel.48.2.281' }
    ]
  },

  {
    id: 'finite_sample_rdd',
    title: 'Finite Sample Inference in Regression Discontinuity Designs',
    tier: 'Advanced',
    description: 'Robust inference, permutation tests, and finite sample corrections for RDD',
    content: `*Content pending. This topic will cover: finite sample inference in RDD, permutation tests, robust variance estimation, coverage properties, and sensitivity to bandwidth choice.*`,
    prerequisites: ['regression_discontinuity'],
    learningObjectives: [
      'Apply finite sample inference methods in RDD',
      'Implement permutation and randomization tests',
      'Conduct robust variance estimation',
      'Assess sensitivity to bandwidth selection'
    ],
    keyDefinitions: [
      { term: 'Finite sample inference', definition: 'Inference methods valid for small sample sizes' },
      { term: 'Permutation test', definition: 'Randomization-based inference without asymptotic approximations' },
      { term: 'Coverage probability', definition: 'Actual frequency confidence intervals contain true parameter' }
    ],
    examples: {
      python: `# Placeholder - finite sample RDD inference`,
      r: `# Placeholder - finite sample RDD inference`
    },
    references: [
      { authors: 'Cattaneo MD et al', title: 'A practical introduction to RD designs', year: 2020, doi: '10.1017/9781108684606' },
      { authors: 'Canay IA, Kamat V', title: 'Approximate permutation tests', year: 2018, doi: '10.1093/restud/rdx062' }
    ]
  },

  {
    id: 'balancing_estimators',
    title: 'Balancing Estimators and Synthetic Controls',
    tier: 'Advanced',
    description: 'Covariate balancing weights, entropy balancing, synthetic control methods',
    content: `*Content pending. This topic will cover: covariate balancing propensity score, entropy balancing, calibration, synthetic controls, matrix completion.*`,
    prerequisites: ['unconfoundedness_propensity_score'],
    learningObjectives: [
      'Apply covariate balancing methods',
      'Implement entropy balancing',
      'Use synthetic control methods',
      'Compare balancing approaches'
    ],
    keyDefinitions: [
      { term: 'Covariate balance', definition: 'Treatment groups have similar covariate distributions' },
      { term: 'Entropy balancing', definition: 'Weights that exactly balance moments while minimizing entropy' },
      { term: 'Synthetic control', definition: 'Weighted average of controls matching treated unit' }
    ],
    examples: {
      python: `# Placeholder - balancing methods`,
      r: `# Placeholder - balancing methods`
    },
    references: [
      { authors: 'Hainmueller J', title: 'Entropy balancing', year: 2012, doi: '10.1093/pan/mpr025' },
      { authors: 'Abadie A et al', title: 'Synthetic control methods', year: 2010, doi: '10.1198/jasa.2009.ap08746' }
    ]
  },

  {
    id: 'panel_data_methods',
    title: 'Methods for Panel Data',
    tier: 'Advanced',
    description: 'Difference-in-differences, fixed effects, event studies, and panel methods',
    content: `*Content pending. This topic will cover: difference-in-differences, two-way fixed effects, event studies, parallel trends, staggered adoption.*`,
    prerequisites: ['intro_causal_inference'],
    learningObjectives: [
      'Apply difference-in-differences',
      'Implement two-way fixed effects',
      'Conduct event studies',
      'Address violations of parallel trends'
    ],
    keyDefinitions: [
      { term: 'DiD', definition: 'Difference-in-differences - comparing changes over time between groups' },
      { term: 'Parallel trends', definition: 'Treated and control would have same trend absent treatment' },
      { term: 'TWFE', definition: 'Two-way fixed effects - unit and time fixed effects' }
    ],
    examples: {
      python: `# Placeholder - DiD analysis`,
      r: `# Placeholder - DiD analysis`
    },
    references: [
      { authors: 'Callaway B, Sant\'Anna PHC', title: 'Difference-in-differences with multiple time periods', year: 2021, doi: '10.1016/j.jeconom.2020.12.001' }
    ]
  },

  {
    id: 'policy_learning',
    title: 'Policy Learning and Optimal Treatment Regimes',
    tier: 'Frontier',
    description: 'Learning optimal individualized treatment rules from data',
    content: `*Content pending. This topic will cover: optimal treatment regimes, value functions, Q-learning, policy tree methods, doubly robust policy learning.*`,
    prerequisites: ['effect_modification'],
    learningObjectives: [
      'Define optimal treatment regimes',
      'Estimate value functions',
      'Apply policy tree methods',
      'Implement doubly robust policy learners'
    ],
    keyDefinitions: [
      { term: 'Treatment regime', definition: 'Rule assigning treatment based on individual characteristics' },
      { term: 'Value function', definition: 'Expected outcome under a treatment regime' },
      { term: 'Policy tree', definition: 'Interpretable tree-based treatment rule' }
    ],
    examples: {
      python: `# Placeholder - policy learning`,
      r: `# Placeholder - policy learning`
    },
    references: [
      { authors: 'Athey S, Wager S', title: 'Policy learning with observational data', year: 2021, doi: '10.3982/ECTA15732' }
    ]
  },

  {
    id: 'dynamic_policies',
    title: 'Evaluating Dynamic Treatment Policies',
    tier: 'Frontier',
    description: 'Dynamic regimes, Q-learning, A-learning, and reinforcement learning for causal inference',
    content: `*Content pending. This topic will cover: dynamic treatment regimes, Q-learning, A-learning, optimal regime estimation, contextual bandits.*`,
    prerequisites: ['g_methods_longitudinal'],
    learningObjectives: [
      'Specify dynamic treatment regimes',
      'Apply Q-learning and A-learning',
      'Estimate optimal dynamic policies',
      'Connect RL and causal inference'
    ],
    keyDefinitions: [
      { term: 'Dynamic regime', definition: 'Time-varying treatment rule based on evolving history' },
      { term: 'Q-learning', definition: 'Backward induction approach to estimating optimal regimes' },
      { term: 'A-learning', definition: 'Advantage learning for optimal regime estimation' }
    ],
    examples: {
      python: `# Placeholder - dynamic regime estimation`,
      r: `# Placeholder - dynamic regime estimation`
    },
    references: [
      { authors: 'Murphy SA', title: 'Optimal dynamic treatment regimes', year: 2003, doi: '10.1111/1467-9868.00389' }
    ]
  },

  {
    id: 'structural_equation_modeling',
    title: 'Structural Equation Modeling',
    tier: 'Advanced',
    description: 'SEM for causal inference, path analysis, and latent variable models',
    content: `*Content pending. This topic will cover: SEM framework, path analysis, measurement models, identification in SEMs, causal interpretation of SEM.*`,
    prerequisites: ['framework_scm'],
    learningObjectives: [
      'Specify structural equation models',
      'Conduct path analysis',
      'Incorporate latent variables',
      'Identify causal effects in SEMs'
    ],
    keyDefinitions: [
      { term: 'SEM', definition: 'System of equations relating observed and latent variables' },
      { term: 'Path analysis', definition: 'Decomposing effects through directed paths' },
      { term: 'Latent variable', definition: 'Unobserved variable inferred from indicators' }
    ],
    examples: {
      python: `# Placeholder - SEM estimation`,
      r: `# Placeholder - SEM estimation`
    },
    references: [
      { authors: 'Bollen KA', title: 'Structural Equations with Latent Variables', year: 1989, doi: '10.1002/9781118619179' }
    ]
  },

  {
    id: 'adaptive_experiments',
    title: 'Adaptive Experiments and Sequential Testing',
    tier: 'Frontier',
    description: 'Adaptive randomization, sequential monitoring, and multi-armed bandits',
    content: `*Content pending. This topic will cover: adaptive designs, response-adaptive randomization, group sequential designs, multi-armed bandits, Thompson sampling.*`,
    prerequisites: ['randomized_experiments'],
    learningObjectives: [
      'Design adaptive experiments',
      'Apply response-adaptive randomization',
      'Implement group sequential methods',
      'Use bandit algorithms for experiments'
    ],
    keyDefinitions: [
      { term: 'Adaptive design', definition: 'Trial design allowing modifications based on accumulating data' },
      { term: 'Multi-armed bandit', definition: 'Sequential decision problem balancing exploration and exploitation' },
      { term: 'Thompson sampling', definition: 'Bayesian approach to adaptive allocation' }
    ],
    examples: {
      python: `# Placeholder - adaptive experiment`,
      r: `# Placeholder - adaptive experiment`
    },
    references: [
      { authors: 'Villar SS et al', title: 'Multi-armed bandit models for clinical trials', year: 2015, doi: '10.1177/1740774515588375' }
    ]
  }
];
