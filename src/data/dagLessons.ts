import { Lesson } from './lessons';

export const dagLessons: Lesson[] = [
  {
    id: 'dag-chain',
    title: 'DAG: Simple Chain',
    description: 'Understanding causal chains X → Y → Z',
    category: 'theory',
    tier: 'Foundational',
    pythonCode: `# Simple Causal Chain: X → Y → Z
# X causes Y, which in turn causes Z

def dag(nodes, edges):
    """Create and display a DAG"""
    print("DAG: Simple Chain")
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("\\nInterpretation:")
    print("- X directly causes Y")
    print("- Y directly causes Z")
    print("- X has an indirect effect on Z through Y")
    return {"nodes": nodes, "edges": edges}

# Define the chain structure
nodes = ['X', 'Y', 'Z']
edges = [('X', 'Y'), ('Y', 'Z')]

dag(nodes=nodes, edges=edges)
`,
    rCode: `# Simple Causal Chain: X → Y → Z
# X causes Y, which in turn causes Z

dag <- function(nodes, edges) {
  cat("DAG: Simple Chain\\n")
  cat("Nodes:", nodes, "\\n")
  cat("Edges:", paste(sapply(edges, function(e) paste(e[1], "->", e[2])), collapse=", "), "\\n")
  cat("\\nInterpretation:\\n")
  cat("- X directly causes Y\\n")
  cat("- Y directly causes Z\\n")
  cat("- X has an indirect effect on Z through Y\\n")
  list(nodes=nodes, edges=edges)
}

# Define the chain structure
nodes <- c('X', 'Y', 'Z')
edges <- list(c('X', 'Y'), c('Y', 'Z'))

dag(nodes=nodes, edges=edges)
`,
    learningObjectives: [
      'Understand direct and indirect causal effects',
      'Identify mediation paths',
      'Recognize transitivity in causation'
    ]
  },
  {
    id: 'dag-fork',
    title: 'DAG: Fork (Common Cause)',
    description: 'Confounding through a common cause: Z → X and Z → Y',
    category: 'theory',
    tier: 'Foundational',
    pythonCode: `# Fork/Common Cause: Z → X and Z → Y
# Z is a confounder that affects both X and Y

def dag(nodes, edges):
    """Create and display a DAG with common cause"""
    print("DAG: Fork (Common Cause)")
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("\\nInterpretation:")
    print("- Z causes both X and Y")
    print("- Z is a CONFOUNDER")
    print("- Must control for Z to identify X→Y effect")
    print("- Without controlling Z, X and Y appear correlated")
    return {"nodes": nodes, "edges": edges}

# Define the fork structure
nodes = ['X', 'Y', 'Z']
edges = [('Z', 'X'), ('Z', 'Y')]

dag(nodes=nodes, edges=edges)
`,
    rCode: `# Fork/Common Cause: Z → X and Z → Y
# Z is a confounder that affects both X and Y

dag <- function(nodes, edges) {
  cat("DAG: Fork (Common Cause)\\n")
  cat("Nodes:", nodes, "\\n")
  cat("Edges:", paste(sapply(edges, function(e) paste(e[1], "->", e[2])), collapse=", "), "\\n")
  cat("\\nInterpretation:\\n")
  cat("- Z causes both X and Y\\n")
  cat("- Z is a CONFOUNDER\\n")
  cat("- Must control for Z to identify X→Y effect\\n")
  cat("- Without controlling Z, X and Y appear correlated\\n")
  list(nodes=nodes, edges=edges)
}

# Define the fork structure
nodes <- c('X', 'Y', 'Z')
edges <- list(c('Z', 'X'), c('Z', 'Y'))

dag(nodes=nodes, edges=edges)
`,
    learningObjectives: [
      'Identify confounders',
      'Understand spurious correlation',
      'Learn when to control for variables'
    ]
  },
  {
    id: 'dag-collider',
    title: 'DAG: Collider (Common Effect)',
    description: 'Selection bias through a common effect: X → Z ← Y',
    category: 'theory',
    tier: 'Intermediate',
    pythonCode: `# Collider/Common Effect: X → Z ← Y
# Z is caused by both X and Y

def dag(nodes, edges):
    """Create and display a DAG with collider"""
    print("DAG: Collider (Common Effect)")
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("\\nInterpretation:")
    print("- Both X and Y cause Z")
    print("- Z is a COLLIDER")
    print("- X and Y are independent")
    print("- DO NOT control for Z!")
    print("- Controlling Z creates spurious association between X and Y")
    print("- This is called 'collider bias' or 'selection bias'")
    return {"nodes": nodes, "edges": edges}

# Define the collider structure
nodes = ['X', 'Y', 'Z']
edges = [('X', 'Z'), ('Y', 'Z')]

dag(nodes=nodes, edges=edges)
`,
    rCode: `# Collider/Common Effect: X → Z ← Y
# Z is caused by both X and Y

dag <- function(nodes, edges) {
  cat("DAG: Collider (Common Effect)\\n")
  cat("Nodes:", nodes, "\\n")
  cat("Edges:", paste(sapply(edges, function(e) paste(e[1], "->", e[2])), collapse=", "), "\\n")
  cat("\\nInterpretation:\\n")
  cat("- Both X and Y cause Z\\n")
  cat("- Z is a COLLIDER\\n")
  cat("- X and Y are independent\\n")
  cat("- DO NOT control for Z!\\n")
  cat("- Controlling Z creates spurious association between X and Y\\n")
  cat("- This is called 'collider bias' or 'selection bias'\\n")
  list(nodes=nodes, edges=edges)
}

# Define the collider structure
nodes <- c('X', 'Y', 'Z')
edges <- list(c('X', 'Z'), c('Y', 'Z'))

dag(nodes=nodes, edges=edges)
`,
    learningObjectives: [
      'Identify colliders',
      'Understand selection bias',
      'Learn when NOT to control for variables'
    ]
  },
  {
    id: 'dag-complex',
    title: 'DAG: Complex Structure',
    description: 'Multiple confounders, mediators, and colliders',
    category: 'theory',
    tier: 'Advanced',
    pythonCode: `# Complex DAG with multiple causal structures
# Treatment (T) → Outcome (Y)
# Confounder (C) → T and C → Y
# Mediator: T → M → Y
# Collider: T → S ← Y

def dag(nodes, edges):
    """Create and display a complex DAG"""
    print("DAG: Complex Causal Structure")
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("\\nInterpretation:")
    print("- T (Treatment) affects Y (Outcome)")
    print("- C is a CONFOUNDER (control for it)")
    print("- M is a MEDIATOR (part of T's effect on Y)")
    print("- S is a COLLIDER (don't control for it)")
    print("\\nEstimation Strategy:")
    print("- Control for C to remove confounding")
    print("- Don't control for M (would block mediation)")
    print("- Don't control for S (would induce collider bias)")
    return {"nodes": nodes, "edges": edges}

# Define complex structure
nodes = ['T', 'Y', 'C', 'M', 'S']
edges = [
    ('T', 'Y'),   # Direct effect
    ('C', 'T'),   # Confounding
    ('C', 'Y'),   # Confounding
    ('T', 'M'),   # Mediation
    ('M', 'Y'),   # Mediation
    ('T', 'S'),   # Collider
    ('Y', 'S')    # Collider
]

dag(nodes=nodes, edges=edges)
`,
    rCode: `# Complex DAG with multiple causal structures
# Treatment (T) → Outcome (Y)
# Confounder (C) → T and C → Y
# Mediator: T → M → Y
# Collider: T → S ← Y

dag <- function(nodes, edges) {
  cat("DAG: Complex Causal Structure\\n")
  cat("Nodes:", nodes, "\\n")
  cat("Edges:", paste(sapply(edges, function(e) paste(e[1], "->", e[2])), collapse=", "), "\\n")
  cat("\\nInterpretation:\\n")
  cat("- T (Treatment) affects Y (Outcome)\\n")
  cat("- C is a CONFOUNDER (control for it)\\n")
  cat("- M is a MEDIATOR (part of T's effect on Y)\\n")
  cat("- S is a COLLIDER (don't control for it)\\n")
  cat("\\nEstimation Strategy:\\n")
  cat("- Control for C to remove confounding\\n")
  cat("- Don't control for M (would block mediation)\\n")
  cat("- Don't control for S (would induce collider bias)\\n")
  list(nodes=nodes, edges=edges)
}

# Define complex structure
nodes <- c('T', 'Y', 'C', 'M', 'S')
edges <- list(
  c('T', 'Y'),   # Direct effect
  c('C', 'T'),   # Confounding
  c('C', 'Y'),   # Confounding
  c('T', 'M'),   # Mediation
  c('M', 'Y'),   # Mediation
  c('T', 'S'),   # Collider
  c('Y', 'S')    # Collider
)

dag(nodes=nodes, edges=edges)
`,
    learningObjectives: [
      'Analyze complex causal structures',
      'Determine adjustment sets',
      'Avoid multiple biases simultaneously'
    ]
  },
  {
    id: 'dag-backdoor',
    title: 'DAG: Backdoor Path',
    description: 'Blocking backdoor paths to identify causal effects',
    category: 'theory',
    tier: 'Advanced',
    pythonCode: `# Backdoor Path Example
# X → Y (causal path of interest)
# X ← Z → Y (backdoor path through Z)

def dag(nodes, edges):
    """Create and display backdoor path DAG"""
    print("DAG: Backdoor Path")
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("\\nCausal Paths:")
    print("- FRONT DOOR: X → Y (what we want to measure)")
    print("- BACKDOOR: X ← Z → Y (confounding path)")
    print("\\nBackdoor Criterion:")
    print("- Control for Z to block the backdoor path")
    print("- This identifies the causal effect X → Y")
    print("- Z satisfies the backdoor criterion")
    return {"nodes": nodes, "edges": edges}

# Define backdoor structure
nodes = ['X', 'Y', 'Z', 'W']
edges = [
    ('X', 'Y'),   # Front door (causal)
    ('Z', 'X'),   # Backdoor
    ('Z', 'Y'),   # Backdoor
    ('W', 'Z')    # Ancestor of confounder
]

dag(nodes=nodes, edges=edges)

print("\\nBackdoor adjustment set: {Z}")
print("Conditioning on W alone is not sufficient!")
`,
    rCode: `# Backdoor Path Example
# X → Y (causal path of interest)
# X ← Z → Y (backdoor path through Z)

dag <- function(nodes, edges) {
  cat("DAG: Backdoor Path\\n")
  cat("Nodes:", nodes, "\\n")
  cat("Edges:", paste(sapply(edges, function(e) paste(e[1], "->", e[2])), collapse=", "), "\\n")
  cat("\\nCausal Paths:\\n")
  cat("- FRONT DOOR: X → Y (what we want to measure)\\n")
  cat("- BACKDOOR: X ← Z → Y (confounding path)\\n")
  cat("\\nBackdoor Criterion:\\n")
  cat("- Control for Z to block the backdoor path\\n")
  cat("- This identifies the causal effect X → Y\\n")
  cat("- Z satisfies the backdoor criterion\\n")
  list(nodes=nodes, edges=edges)
}

# Define backdoor structure
nodes <- c('X', 'Y', 'Z', 'W')
edges <- list(
  c('X', 'Y'),   # Front door (causal)
  c('Z', 'X'),   # Backdoor
  c('Z', 'Y'),   # Backdoor
  c('W', 'Z')    # Ancestor of confounder
)

dag(nodes=nodes, edges=edges)

cat("\\nBackdoor adjustment set: {Z}\\n")
cat("Conditioning on W alone is not sufficient!\\n")
`,
    learningObjectives: [
      'Identify backdoor paths',
      'Apply backdoor criterion',
      'Find valid adjustment sets'
    ]
  }
];
