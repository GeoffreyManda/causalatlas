/**
 * Code Translation Service
 * Translates code between Python, R, JavaScript, and HTML
 * Uses pattern matching and AST-like transformations for statistical/causal inference code
 */

export type Language = 'python' | 'r' | 'javascript' | 'html';

interface TranslationPattern {
  source: RegExp;
  target: string | ((match: RegExpMatchArray) => string);
}

interface LanguagePair {
  from: Language;
  to: Language;
  patterns: TranslationPattern[];
}

// Python to R translation patterns
const pythonToR: TranslationPattern[] = [
  // Imports
  { source: /import numpy as np/g, target: '# numpy operations use base R or Matrix package' },
  { source: /import pandas as pd/g, target: 'library(dplyr)' },
  { source: /from sklearn\.linear_model import (.*)/g, target: (m) => `# ${m[1]}: use lm() or glm()` },
  { source: /from sklearn\.ensemble import (.*)/g, target: (m) => `library(randomForest) # for ${m[1]}` },
  { source: /import matplotlib\.pyplot as plt/g, target: '# use base plot() or ggplot2' },

  // Variable assignment
  { source: /(\w+)\s*=\s*np\.random\.seed\((\d+)\)/g, target: 'set.seed($2)' },
  { source: /(\w+)\s*=\s*np\.random\.normal\(size=\((\d+),\s*(\d+)\)\)/g, target: '$1 <- matrix(rnorm($2*$3), $2, $3)' },
  { source: /(\w+)\s*=\s*np\.random\.normal\(size=(\d+)\)/g, target: '$1 <- rnorm($2)' },
  { source: /(\w+)\s*=\s*np\.random\.binomial\((\d+),\s*([^,\)]+)(?:,\s*(\d+))?\)/g, target: (m) => m[4] ? `$1 <- rbinom(${m[4]}, ${m[2]}, ${m[3]})` : `$1 <- rbinom(${m[2]}, 1, ${m[3]})` },

  // Array operations
  { source: /np\.array\((.*?)\)/g, target: 'c($1)' },
  { source: /np\.zeros\((\d+)\)/g, target: 'rep(0, $1)' },
  { source: /np\.ones\((\d+)\)/g, target: 'rep(1, $1)' },
  { source: /\.reshape\((-?\d+),\s*(-?\d+)\)/g, target: '' }, // R handles this implicitly
  { source: /\.shape\[0\]/g, target: 'nrow' },
  { source: /\.shape\[1\]/g, target: 'ncol' },
  { source: /len\((.*?)\)/g, target: 'length($1)' },

  // Indexing (Python 0-based to R 1-based)
  { source: /\[(\w+)==0\]/g, target: '[$1==0,]' },
  { source: /\[(\w+)==1\]/g, target: '[$1==1,]' },
  { source: /\[:,\s*(\d+)\]/g, target: (m) => `[,${parseInt(m[1]) + 1}]` },
  { source: /\[(\w+),:\]/g, target: '[$1,]' },

  // Math operations
  { source: /np\.exp\((.*?)\)/g, target: 'exp($1)' },
  { source: /np\.log\((.*?)\)/g, target: 'log($1)' },
  { source: /np\.mean\((.*?)\)/g, target: 'mean($1)' },
  { source: /np\.sum\((.*?)\)/g, target: 'sum($1)' },
  { source: /\.mean\(\)/g, target: '' }, // handled by mean() wrapper
  { source: /\.sum\(\)/g, target: '' }, // handled by sum() wrapper
  { source: /\.std\(\)/g, target: '' }, // will use sd() wrapper

  // Statistical models
  { source: /LogisticRegression\(([^)]*)\)\.fit\(([^,]+),\s*([^)]+)\)/g, target: 'glm($3 ~ $2, family=binomial)' },
  { source: /LinearRegression\(\)\.fit\(([^,]+),\s*([^)]+)\)/g, target: 'lm($2 ~ $1)' },
  { source: /\.predict\((.*?)\)/g, target: 'predict(model, newdata=$1)' },
  { source: /\.predict_proba\((.*?)\)\[:,\s*1\]/g, target: 'predict(model, newdata=$1, type="response")' },

  // Control flow
  { source: /if\s+(.*?):/g, target: 'if ($1) {' },
  { source: /elif\s+(.*?):/g, target: '} else if ($1) {' },
  { source: /else:/g, target: '} else {' },
  { source: /for\s+(\w+)\s+in\s+range\((\d+)\):/g, target: 'for ($1 in 1:$2) {' },

  // Printing
  { source: /print\(f"([^"]*?)\{([^}]+)\}([^"]*?)"\)/g, target: (m) => `cat("${m[1]}", ${m[2]}, "${m[3]}", "\\n")` },
  { source: /print\((.*?)\)/g, target: 'print($1)' },

  // Syntax
  { source: /True/g, target: 'TRUE' },
  { source: /False/g, target: 'FALSE' },
  { source: /None/g, target: 'NULL' },
  { source: /(\w+)\s*=\s*/g, target: '$1 <- ' }, // Assignment
];

// R to Python translation patterns
const rToPython: TranslationPattern[] = [
  // Imports
  { source: /library\((.*?)\)/g, target: (m) => {
    const lib = m[1];
    if (lib === 'dplyr') return 'import pandas as pd';
    if (lib === 'randomForest') return 'from sklearn.ensemble import RandomForestRegressor';
    return `# R package ${lib} - find Python equivalent`;
  }},

  // Random seed
  { source: /set\.seed\((\d+)\)/g, target: 'np.random.seed($1)' },

  // Random distributions
  { source: /matrix\(rnorm\((\d+)\*(\d+)\),\s*(\d+),\s*(\d+)\)/g, target: 'np.random.normal(size=($3, $4))' },
  { source: /rnorm\((\d+)\)/g, target: 'np.random.normal(size=$1)' },
  { source: /rbinom\((\d+),\s*1,\s*([^)]+)\)/g, target: 'np.random.binomial(1, $2, $1)' },
  { source: /runif\((\d+)\)/g, target: 'np.random.uniform(size=$1)' },

  // Basic operations
  { source: /rep\(0,\s*(\d+)\)/g, target: 'np.zeros($1)' },
  { source: /rep\(1,\s*(\d+)\)/g, target: 'np.ones($1)' },
  { source: /c\((.*?)\)/g, target: 'np.array([$1])' },
  { source: /length\((.*?)\)/g, target: 'len($1)' },
  { source: /nrow\((.*?)\)/g, target: '$1.shape[0]' },
  { source: /ncol\((.*?)\)/g, target: '$1.shape[1]' },

  // Indexing (R 1-based to Python 0-based)
  { source: /\[,(\d+)\]/g, target: (m) => `[:, ${parseInt(m[1]) - 1}]` },
  { source: /\[(\w+)==0,\]/g, target: '[$1==0]' },
  { source: /\[(\w+)==1,\]/g, target: '[$1==1]' },

  // Math operations
  { source: /plogis\((.*?)\)/g, target: '1/(1+np.exp(-($1)))' },
  { source: /qnorm\((.*?)\)/g, target: 'scipy.stats.norm.ppf($1)' },

  // Statistical models
  { source: /glm\(([^~]+)\s*~\s*([^,]+),\s*family\s*=\s*binomial\)/g, target: 'LogisticRegression().fit($2, $1)' },
  { source: /lm\(([^~]+)\s*~\s*([^)]+)\)/g, target: 'LinearRegression().fit($2, $1)' },
  { source: /predict\(([^,]+),\s*newdata\s*=\s*([^,)]+)(?:,\s*type\s*=\s*"response")?\)/g, target: '$1.predict($2)' },

  // Control flow
  { source: /if\s*\((.*?)\)\s*\{/g, target: 'if $1:' },
  { source: /\}\s*else\s*if\s*\((.*?)\)\s*\{/g, target: '\nelif $1:' },
  { source: /\}\s*else\s*\{/g, target: '\nelse:' },
  { source: /for\s*\((\w+)\s+in\s+(\d+):(\d+)\)\s*\{/g, target: 'for $1 in range($2-1, $3):' },
  { source: /\}/g, target: '' }, // Python uses indentation

  // Printing
  { source: /cat\((.*?)(?:,\s*"\\n")?\)/g, target: 'print($1)' },

  // Syntax
  { source: /TRUE/g, target: 'True' },
  { source: /FALSE/g, target: 'False' },
  { source: /NULL/g, target: 'None' },
  { source: /(\w+)\s*<-\s*/g, target: '$1 = ' }, // Assignment
];

// Python to JavaScript translation
const pythonToJS: TranslationPattern[] = [
  // Imports (convert to require/import)
  { source: /import numpy as np/g, target: '// Use math-js or numeric.js for numpy-like operations' },
  { source: /import pandas as pd/g, target: '// Use dataframe-js or danfo.js' },

  // Random operations
  { source: /np\.random\.seed\((\d+)\)/g, target: '// Set Math.random seed (use seedrandom library)' },
  { source: /np\.random\.normal\(size=(\d+)\)/g, target: 'Array.from({length: $1}, () => normalRandom())' },
  { source: /np\.random\.binomial\(1,\s*([^,)]+),\s*(\d+)\)/g, target: 'Array.from({length: $2}, () => Math.random() < $1 ? 1 : 0)' },

  // Array operations
  { source: /np\.array\((.*?)\)/g, target: '[$1]' },
  { source: /np\.zeros\((\d+)\)/g, target: 'Array($1).fill(0)' },
  { source: /np\.ones\((\d+)\)/g, target: 'Array($1).fill(1)' },
  { source: /\.mean\(\)/g, target: '.reduce((a,b) => a+b) / array.length' },
  { source: /\.sum\(\)/g, target: '.reduce((a,b) => a+b, 0)' },
  { source: /len\((.*?)\)/g, target: '$1.length' },

  // Control flow (similar to Python)
  { source: /print\(f"([^"]*?)\{([^}]+)\}([^"]*?)"\)/g, target: 'console.log(`$1${$2}$3`)' },
  { source: /print\((.*?)\)/g, target: 'console.log($1)' },

  // Syntax
  { source: /True/g, target: 'true' },
  { source: /False/g, target: 'false' },
  { source: /None/g, target: 'null' },
  { source: /elif/g, target: 'else if' },
];

// JavaScript to Python translation
const jsToPython: TranslationPattern[] = [
  // Random operations
  { source: /Math\.random\(\)/g, target: 'np.random.uniform()' },
  { source: /Array\.from\(\{length:\s*(\d+)\},\s*\(\)\s*=>\s*(.*?)\)/g, target: 'np.array([$2 for _ in range($1)])' },
  { source: /Array\((\d+)\)\.fill\(0\)/g, target: 'np.zeros($1)' },
  { source: /Array\((\d+)\)\.fill\(1\)/g, target: 'np.ones($1)' },

  // Array operations
  { source: /\.reduce\(\(a,\s*b\)\s*=>\s*a\s*\+\s*b,\s*0\)/g, target: '.sum()' },
  { source: /\.reduce\(\(a,\s*b\)\s*=>\s*a\s*\+\s*b\)\s*\/\s*\w+\.length/g, target: '.mean()' },
  { source: /\.length/g, target: 'len()' },

  // Console
  { source: /console\.log\((.*?)\)/g, target: 'print($1)' },

  // Syntax
  { source: /true/g, target: 'True' },
  { source: /false/g, target: 'False' },
  { source: /null/g, target: 'None' },
  { source: /const\s+(\w+)\s*=/g, target: '$1 =' },
  { source: /let\s+(\w+)\s*=/g, target: '$1 =' },
  { source: /var\s+(\w+)\s*=/g, target: '$1 =' },
];

// HTML template conversions
const pythonToHTML: TranslationPattern[] = [
  { source: /^.*$/gm, target: (m) => {
    return `<!DOCTYPE html>
<html>
<head>
  <title>Python to HTML Conversion</title>
  <script src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js"></script>
</head>
<body>
  <h1>Python Code Execution</h1>
  <pre id="output"></pre>
  <script>
    async function runPython() {
      let pyodide = await loadPyodide();
      let result = pyodide.runPython(\`
${m[0]}
      \`);
      document.getElementById('output').textContent = result;
    }
    runPython();
  </script>
</body>
</html>`;
  }}
];

const rToHTML: TranslationPattern[] = [
  { source: /^.*$/gm, target: (m) => {
    return `<!DOCTYPE html>
<html>
<head>
  <title>R to HTML Conversion</title>
  <script src="https://cdn.jsdelivr.net/npm/webr@0.3.3/dist/webr.js"></script>
</head>
<body>
  <h1>R Code Execution</h1>
  <pre id="output"></pre>
  <script type="module">
    const { WebR } = await import('https://cdn.jsdelivr.net/npm/webr@0.3.3/dist/webr.mjs');
    const webR = new WebR();
    await webR.init();
    const result = await webR.evalRString(\`
${m[0]}
    \`);
    document.getElementById('output').textContent = result;
  </script>
</body>
</html>`;
  }}
];

// Build translation map
const translationMap: Map<string, TranslationPattern[]> = new Map([
  ['python-r', pythonToR],
  ['r-python', rToPython],
  ['python-javascript', pythonToJS],
  ['javascript-python', jsToPython],
  ['python-html', pythonToHTML],
  ['r-html', rToHTML],
]);

/**
 * Translate code from one language to another
 */
export function translateCode(code: string, from: Language, to: Language): string {
  if (from === to) return code;

  const key = `${from}-${to}`;
  const patterns = translationMap.get(key);

  if (!patterns) {
    return `// Translation from ${from} to ${to} not yet implemented\n\n${code}`;
  }

  let translated = code;

  // Apply each pattern transformation
  for (const pattern of patterns) {
    if (typeof pattern.target === 'string') {
      translated = translated.replace(pattern.source, pattern.target);
    } else {
      translated = translated.replace(pattern.source, pattern.target as any);
    }
  }

  // Post-processing based on target language
  if (to === 'python') {
    translated = addPythonImports(translated);
  } else if (to === 'r') {
    translated = addRLibraries(translated);
  } else if (to === 'javascript') {
    translated = addJavaScriptHelpers(translated);
  }

  return translated;
}

/**
 * Detect necessary Python imports from code
 */
function addPythonImports(code: string): string {
  const imports = new Set<string>();

  if (/np\.|numpy/.test(code)) imports.add('import numpy as np');
  if (/pd\.|pandas/.test(code)) imports.add('import pandas as pd');
  if (/LogisticRegression|LinearRegression|RandomForest/.test(code)) imports.add('from sklearn.linear_model import LogisticRegression, LinearRegression');
  if (/scipy/.test(code)) imports.add('import scipy.stats');

  if (imports.size > 0) {
    return Array.from(imports).join('\n') + '\n\n' + code;
  }

  return code;
}

/**
 * Detect necessary R libraries from code
 */
function addRLibraries(code: string): string {
  const libraries = new Set<string>();

  if (/dplyr|mutate|filter|select/.test(code)) libraries.add('library(dplyr)');
  if (/ggplot/.test(code)) libraries.add('library(ggplot2)');
  if (/randomForest/.test(code)) libraries.add('library(randomForest)');

  if (libraries.size > 0) {
    return Array.from(libraries).join('\n') + '\n\n' + code;
  }

  return code;
}

/**
 * Add JavaScript helper functions if needed
 */
function addJavaScriptHelpers(code: string): string {
  let helpers = '';

  if (/normalRandom/.test(code)) {
    helpers += `// Box-Muller transform for normal random numbers
function normalRandom() {
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

`;
  }

  return helpers + code;
}

/**
 * Get supported translation pairs
 */
export function getSupportedTranslations(): Array<{from: Language; to: Language}> {
  return [
    { from: 'python', to: 'r' },
    { from: 'r', to: 'python' },
    { from: 'python', to: 'javascript' },
    { from: 'javascript', to: 'python' },
    { from: 'python', to: 'html' },
    { from: 'r', to: 'html' },
  ];
}
