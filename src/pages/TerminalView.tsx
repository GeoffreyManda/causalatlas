import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Play, Trash2, Copy, Check } from 'lucide-react';
import { toast } from 'sonner';

const TerminalView = () => {
  const location = useLocation();
  const [pythonCode, setPythonCode] = useState('');
  const [rCode, setRCode] = useState('');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState('python');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (location.state?.code && location.state?.language) {
      const { code, language } = location.state;
      if (language === 'Python') {
        setPythonCode(code);
        setActiveTab('python');
      } else if (language === 'R') {
        setRCode(code);
        setActiveTab('r');
      }
    }
  }, [location.state]);

  const runCode = async () => {
    setIsRunning(true);
    setOutput('');
    
    const code = activeTab === 'python' ? pythonCode : rCode;
    
    if (!code.trim()) {
      toast.error('Please enter some code to run');
      setIsRunning(false);
      return;
    }

    // Simulate code execution
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setOutput(`=== Executing ${activeTab.toUpperCase()} Code ===\n\n` +
      `Note: Full Python/R execution requires Pyodide/WebR integration.\n` +
      `This demo shows the interface structure.\n\n` +
      `Code preview:\n${code.split('\n').slice(0, 3).join('\n')}${code.split('\n').length > 3 ? '\n...' : ''}\n\n` +
      `[Simulated output would appear here]`);
    
    setIsRunning(false);
    toast.success('Code executed successfully');
  };

  const clearOutput = () => {
    setOutput('');
    toast.info('Output cleared');
  };

  const copyCode = async () => {
    const code = activeTab === 'python' ? pythonCode : rCode;
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success('Code copied to clipboard');
  };

  const examplePython = `import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

# Set seed for reproducibility
np.random.seed(20251111)

# Generate synthetic data
n = 2000
X = np.random.normal(size=(n, 3))
e = 1/(1 + np.exp(-(0.3*X[:,0] - 0.2*X[:,1] + 0.1*X[:,2])))
A = np.random.binomial(1, e)
Y = 2*A + X[:,0] + 0.5*X[:,1] + np.random.normal(size=n)

# AIPW estimation with cross-fitting
idx = np.arange(n)
np.random.shuffle(idx)
folds = [idx[:n//2], idx[n//2:]]

phi = np.zeros(n)
for k in range(2):
    train, test = folds[1-k], folds[k]
    
    # Fit propensity model
    prop = LogisticRegression(max_iter=300).fit(X[train], A[train])
    ehat = prop.predict_proba(X[test])[:,1]
    
    # Fit outcome models
    mu1 = LinearRegression().fit(X[train][A[train]==1], Y[train][A[train]==1])
    mu0 = LinearRegression().fit(X[train][A[train]==0], Y[train][A[train]==0])
    mu1h = mu1.predict(X[test])
    mu0h = mu0.predict(X[test])
    
    # Compute efficient influence function
    Ai, Yi = A[test], Y[test]
    phi[test] = Ai/ehat*(Yi - mu1h) - (1-Ai)/(1-ehat)*(Yi - mu0h) + (mu1h - mu0h)

# Estimate ATE
psi = phi.mean()
se = phi.std(ddof=1) / np.sqrt(n)

print(f"ATE (AIPW, cross-fit): {psi:.3f} ± {1.96*se:.3f}")`;

  const exampleR = `# MSM Estimation with IPW
set.seed(20251111)

n <- 3000

# Longitudinal data structure
L1 <- rnorm(n)
A1 <- rbinom(n, 1, plogis(0.5*L1))
L2 <- 0.4*A1 + rnorm(n)
A2 <- rbinom(n, 1, plogis(0.5*L2 - 0.3*A1))
Y <- 1.2*A1 + 1.5*A2 + 0.5*L1 + 0.3*L2 + rnorm(n)

# Compute stabilized weights
pA1 <- plogis(0.5*L1)
pA2 <- plogis(0.5*L2 - 0.3*A1)
sw <- (plogis(0.5*A1)*plogis(0.5*A2)) / (pA1*pA2)

# Fit MSM
msm <- glm(Y ~ A1 + A2, weights=sw)

print("MSM Coefficients:")
print(coef(msm))`;

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Interactive Terminal</h1>
          <p className="text-muted-foreground">
            Execute Python and R code examples with reproducible seeds
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Code Editor */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Code Editor</h2>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={copyCode}
                  disabled={!pythonCode && !rCode}
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
                <Button
                  size="sm"
                  onClick={runCode}
                  disabled={isRunning}
                  className="gap-2"
                >
                  <Play className="h-4 w-4" />
                  {isRunning ? 'Running...' : 'Run'}
                </Button>
              </div>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="python">Python</TabsTrigger>
                <TabsTrigger value="r">R</TabsTrigger>
              </TabsList>
              
              <TabsContent value="python" className="mt-4">
                <div className="relative">
                  <textarea
                    value={pythonCode}
                    onChange={(e) => setPythonCode(e.target.value)}
                    placeholder="Enter Python code here..."
                    className="w-full h-[500px] p-4 font-mono text-sm bg-terminal-bg text-terminal-fg rounded-lg border focus:outline-none focus:ring-2 focus:ring-accent resize-none"
                  />
                  {!pythonCode && (
                    <Button
                      size="sm"
                      variant="secondary"
                      className="absolute top-4 right-4"
                      onClick={() => setPythonCode(examplePython)}
                    >
                      Load Example
                    </Button>
                  )}
                </div>
              </TabsContent>
              
              <TabsContent value="r" className="mt-4">
                <div className="relative">
                  <textarea
                    value={rCode}
                    onChange={(e) => setRCode(e.target.value)}
                    placeholder="Enter R code here..."
                    className="w-full h-[500px] p-4 font-mono text-sm bg-terminal-bg text-terminal-fg rounded-lg border focus:outline-none focus:ring-2 focus:ring-accent resize-none"
                  />
                  {!rCode && (
                    <Button
                      size="sm"
                      variant="secondary"
                      className="absolute top-4 right-4"
                      onClick={() => setRCode(exampleR)}
                    >
                      Load Example
                    </Button>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </div>

          {/* Output */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Output</h2>
              <Button
                size="sm"
                variant="outline"
                onClick={clearOutput}
                disabled={!output}
                className="gap-2"
              >
                <Trash2 className="h-4 w-4" />
                Clear
              </Button>
            </div>

            <div className="h-[556px] p-4 font-mono text-sm bg-terminal-bg text-terminal-fg rounded-lg border overflow-auto">
              {output ? (
                <pre className="whitespace-pre-wrap">{output}</pre>
              ) : (
                <div className="flex items-center justify-center h-full text-terminal-fg/50">
                  Output will appear here after running code
                </div>
              )}
            </div>

            {/* Info Box */}
            <div className="p-4 rounded-lg border bg-muted/50">
              <h3 className="font-semibold text-sm mb-2">Implementation Note</h3>
              <p className="text-xs text-muted-foreground">
                Full Python/R execution requires Pyodide/WebR integration or a local bridge server. 
                This demo shows the terminal interface structure. To enable full functionality, 
                integrate the Pyodide WASM runtime for client-side Python execution or set up 
                the Flask bridge server for local execution.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TerminalView;
