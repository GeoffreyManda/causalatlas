import { TheoryTopic, causalTheory } from './theory';
import { frameworks } from './frameworks';
import { studyDesigns } from './studyDesigns';
import { estimandFamilies } from './estimandFamilies';

// Convert frameworks to TheoryTopics
const frameworkTopics: TheoryTopic[] = frameworks.map(fw => ({
  id: `framework-${fw.id}`,
  title: fw.title,
  tier: 'Foundational' as const,
  description: fw.description,
  content: fw.content,
  prerequisites: [],
  learningObjectives: fw.whenToUse.slice(0, 4),
  keyDefinitions: fw.keyFeatures.map((feature, idx) => ({
    term: `Feature ${idx + 1}`,
    definition: feature
  })),
  examples: {
    python: `# ${fw.title}\n\n${fw.examples}\n\n# Key features:\n${fw.keyFeatures.map((f, i) => `# ${i + 1}. ${f}`).join('\n')}`,
    r: `# ${fw.title}\n\n${fw.examples}\n\n# Key features:\n${fw.keyFeatures.map((f, i) => `# ${i + 1}. ${f}`).join('\n')}`
  },
  references: [
    { authors: 'Pearl J', title: 'Causality', year: 2009, doi: '10.1017/CBO9780511803161' },
    { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
  ]
}));

// Convert study designs to TheoryTopics
const studyDesignTopics: TheoryTopic[] = studyDesigns.map(design => ({
  id: `design-${design.id}`,
  title: design.title,
  tier: 'Intermediate' as const,
  description: design.description,
  content: design.content,
  prerequisites: [],
  learningObjectives: [
    `Understand when to use ${design.title}`,
    `Recognize strengths and limitations`,
    `Identify appropriate estimands`,
    `Apply correct analysis methods`
  ],
  keyDefinitions: [
    ...design.strengths.slice(0, 3).map((s, idx) => ({ term: `Strength ${idx + 1}`, definition: s })),
    ...design.limitations.slice(0, 2).map((l, idx) => ({ term: `Limitation ${idx + 1}`, definition: l }))
  ],
  examples: {
    python: `# ${design.title}\n\n${design.example}\n\n# Typical estimands:\n${design.typicalEstimands.map((e, i) => `# ${i + 1}. ${e}`).join('\n')}`,
    r: `# ${design.title}\n\n${design.example}\n\n# Typical estimands:\n${design.typicalEstimands.map((e, i) => `# ${i + 1}. ${e}`).join('\n')}`
  },
  references: [
    { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' }
  ]
}));

// Convert estimand families to TheoryTopics
const familyTopics: TheoryTopic[] = estimandFamilies.map(family => ({
  id: `family-${family.id}`,
  title: family.title,
  tier: 'Advanced' as const,
  description: family.description,
  content: family.content,
  prerequisites: [],
  learningObjectives: family.keyQuestions.slice(0, 4),
  keyDefinitions: family.commonEstimands.slice(0, 5).map((est, idx) => ({
    term: est,
    definition: `A key estimand in ${family.title}`
  })),
  examples: {
    python: `# ${family.title}\n\n${family.example}\n\n# Common estimands:\n${family.commonEstimands.map((e, i) => `# ${i + 1}. ${e}`).join('\n')}`,
    r: `# ${family.title}\n\n${family.example}\n\n# Common estimands:\n${family.commonEstimands.map((e, i) => `# ${i + 1}. ${e}`).join('\n')}`
  },
  references: [
    { authors: 'Various', title: `Research on ${family.title}`, year: 2023, doi: '10.1000/example' }
  ]
}));

// Root intro topic
const introTopic: TheoryTopic = {
  id: 'intro-causal-inference',
  title: 'Introduction to Causal Inference',
  tier: 'Foundational',
  description: 'Foundations of causal reasoning: from association to causation',
  content: `Causal inference is the science of determining whether one variable **causes** changes in another. Unlike correlation, which merely describes association, causation implies a mechanism: changing X will change Y.

## Why Causal Inference Matters

Understanding causation is critical for decision-making:
- **Medicine:** Does this drug cure disease?
- **Policy:** Will this program reduce poverty?
- **Business:** Does advertising increase sales?

## Association vs. Causation

Ice cream sales and drowning deaths are correlated, but ice cream doesn't cause drowning. Both are caused by a third factor: warm weather. This illustrates **confounding** - when a third variable affects both treatment and outcome.

## The Gold Standard: Randomized Experiments

In randomized controlled trials (RCTs), treatment is assigned randomly. This ensures that treated and control groups are comparable on all characteristics - both measured and unmeasured. Any difference in outcomes can be attributed to treatment.

## The Fundamental Problem

For any individual, we observe only one potential outcome - either under treatment or under control, never both. The unobserved outcome is a **counterfactual**. Causal inference methods attempt to estimate what would have happened in the counterfactual world.

## Modern Causal Inference

Today's causal inference combines:
- **Formal frameworks:** Potential outcomes, structural causal models, directed acyclic graphs
- **Identification theory:** Conditions under which causal effects can be learned from data
- **Estimation methods:** Statistical and machine learning techniques that leverage identification results
- **Sensitivity analysis:** Assessing robustness to violations of assumptions`,
  prerequisites: [],
  learningObjectives: [
    'Distinguish causation from association',
    'Understand the fundamental problem of causal inference',
    'Recognize the role of randomization',
    'Learn basic causal inference frameworks'
  ],
  keyDefinitions: [
    { term: 'Causation', definition: 'A relationship where changing X will change Y' },
    { term: 'Association', definition: 'A statistical relationship where X and Y occur together' },
    { term: 'Confounding', definition: 'When a third variable affects both treatment and outcome' },
    { term: 'Randomization', definition: 'Random assignment to create comparable groups' },
    { term: 'Counterfactual', definition: 'The hypothetical outcome under a different treatment' }
  ],
  examples: {
    python: `import numpy as np
import pandas as pd

# Simulate confounding
np.random.seed(42)
n = 1000

# Confounder (e.g., motivation)
confounder = np.random.normal(0, 1, n)

# Treatment depends on confounder (selection bias)
treatment = (confounder + np.random.normal(0, 0.5, n) > 0).astype(int)

# Outcome depends on confounder AND treatment
# True causal effect = 2.0
outcome = 5.0 + 2.0 * treatment + 3.0 * confounder + np.random.normal(0, 1, n)

# Naive comparison (biased!)
naive_effect = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()

print(f"True causal effect: 2.0")
print(f"Naive estimate (biased): {naive_effect:.2f}")
print("Bias is due to confounding by motivation!")`,
    r: `set.seed(42)
n <- 1000

# Confounder (e.g., motivation)
confounder <- rnorm(n, 0, 1)

# Treatment depends on confounder
treatment <- as.integer(confounder + rnorm(n, 0, 0.5) > 0)

# Outcome depends on confounder AND treatment
# True causal effect = 2.0
outcome <- 5.0 + 2.0 * treatment + 3.0 * confounder + rnorm(n, 0, 1)

# Naive comparison (biased!)
naive_effect <- mean(outcome[treatment == 1]) - mean(outcome[treatment == 0])

cat("True causal effect: 2.0\\n")
cat("Naive estimate (biased):", round(naive_effect, 2), "\\n")`
  },
  references: [
    { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If', year: 2020, doi: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' },
    { authors: 'Pearl J', title: 'Causality', year: 2009, doi: '10.1017/CBO9780511803161' },
    { authors: 'Holland PW', title: 'Statistics and Causal Inference', year: 1986, doi: '10.1080/01621459.1986.10478354' }
  ]
};

// Export all theory topics combined
export const allTheoryTopics: TheoryTopic[] = [
  introTopic,
  ...causalTheory,  // Include core theory topics (DAGs, identification, etc.)
  ...frameworkTopics,
  ...studyDesignTopics,
  ...familyTopics
];
