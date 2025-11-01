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
    id: 'what_is_causality',
    title: 'What is Causality?',
    tier: 'Foundational',
    description: 'Introduction to causal reasoning vs statistical association',
    content: `Causality answers "what if" questions about interventions. Unlike statistical association (correlation), causality describes what happens when we actively change a variable.

**Key Distinction:**
- Association: X and Y occur together (P(Y|X))
- Causation: Changing X causes Y to change (P(Y|do(X)))

The do-operator represents an intervention that sets X to a value, breaking all incoming arrows to X in a causal graph.`,
    prerequisites: [],
    learningObjectives: [
      'Distinguish causation from association',
      'Understand the do-operator',
      'Recognize confounding'
    ],
    keyDefinitions: [
      { term: 'Intervention', definition: 'An external action that sets a variable to a specific value' },
      { term: 'do-operator', definition: 'do(X=x) represents intervention, written P(Y|do(X=x))' },
      { term: 'Confounding', definition: 'A common cause of treatment and outcome that creates spurious association' }
    ],
    examples: {
      python: `import numpy as np
import matplotlib.pyplot as plt

# Demonstrate confounding: Z causes both X and Y
np.random.seed(42)
n = 1000

# Confounder Z (e.g., age)
Z = np.random.normal(50, 15, n)

# Treatment X depends on Z
X = (Z > 50).astype(int) + np.random.binomial(1, 0.1, n)
X = np.clip(X, 0, 1)

# Outcome Y depends on Z, NOT on X
Y = 2 * Z + np.random.normal(0, 10, n)

# Naive association
print(f"Association E[Y|X=1] - E[Y|X=0]: {Y[X==1].mean() - Y[X==0].mean():.2f}")
print("^ Shows strong 'effect' but X doesn't cause Y!")

# Controlling for Z reveals no causal effect
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(np.c_[X, Z], Y)
print(f"\\nCausal effect (controlling Z): {model.coef_[0]:.2f}")
print("^ Close to 0, revealing no causal relationship")`,
      r: `set.seed(42)
n <- 1000

# Confounder Z
Z <- rnorm(n, 50, 15)

# Treatment X depends on Z
X <- as.integer(Z > 50) + rbinom(n, 1, 0.1)
X <- pmin(X, 1)

# Outcome Y depends on Z, NOT on X
Y <- 2 * Z + rnorm(n, 0, 10)

# Naive association
cat("Association:", mean(Y[X==1]) - mean(Y[X==0]), "\\n")

# Controlling for Z
model <- lm(Y ~ X + Z)
cat("Causal effect (controlling Z):", coef(model)[2], "\\n")`
    },
    references: [
      { authors: 'Pearl J', title: 'Causality: Models, Reasoning, and Inference', year: 2009, doi: '10.1017/CBO9780511803161' }
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
    prerequisites: ['what_is_causality'],
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
  }
];
