import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Play, Trash2, Copy, Check, Package } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { estimandsData } from '@/data/estimands';

const TerminalView = () => {
  const [searchParams] = useSearchParams();
  const [selectedEstimand, setSelectedEstimand] = useState<string>('ate');
  const [pythonCode, setPythonCode] = useState('');
  const [rCode, setRCode] = useState('');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [currentTab, setCurrentTab] = useState('python');
  const [copied, setCopied] = useState(false);
  const [packagesInstalled, setPackagesInstalled] = useState(false);

  // Auto-load code from URL
  useEffect(() => {
    const code = searchParams.get('code');
    const lang = searchParams.get('lang');
    const id = searchParams.get('id');
    
    if (id) {
      setSelectedEstimand(id);
    }
    
    if (code) {
      const decodedCode = decodeURIComponent(code);
      if (lang === 'python') {
        setPythonCode(decodedCode);
        setCurrentTab('python');
      } else if (lang === 'r') {
        setRCode(decodedCode);
        setCurrentTab('r');
      }
    }
  }, [searchParams]);

  // Load estimand code when selection changes
  useEffect(() => {
    const estimand = estimandsData.find(e => e.id === selectedEstimand);
    if (estimand) {
      setPythonCode(estimand.examples.python);
      setRCode(estimand.examples.r);
      setOutput('');
      setPackagesInstalled(false);
    }
  }, [selectedEstimand]);

  const installPackages = async () => {
    setOutput('📦 Installing required packages...\n\n');
    setIsRunning(true);
    
    const requirements = currentTab === 'python' 
      ? ['numpy', 'scipy', 'scikit-learn', 'pandas']
      : ['survival', 'glmnet', 'randomForest', 'ppcor'];
    
    for (const pkg of requirements) {
      await new Promise(resolve => setTimeout(resolve, 300));
      setOutput(prev => prev + `✓ ${pkg}\n`);
    }
    
    setOutput(prev => prev + '\n✅ All packages installed successfully!\n');
    setPackagesInstalled(true);
    setIsRunning(false);
    toast.success('Packages installed');
  };

  const runCode = async () => {
    if (!packagesInstalled) {
      toast.error('Please install packages first');
      return;
    }
    
    setIsRunning(true);
    const code = currentTab === 'python' ? pythonCode : rCode;
    
    if (!code.trim()) {
      toast.error('No code to run');
      setIsRunning(false);
      return;
    }

    setOutput(prev => prev + `\n\n=== Running ${currentTab.toUpperCase()} ===\n\n`);
    
    // Simulate execution
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setOutput(prev => prev + 
      `[Simulated output - integrate Pyodide/WebR for real execution]\n\n` +
      `Code length: ${code.split('\n').length} lines\n` +
      `Estimated runtime: ~1.2s\n\n` +
      `✓ Execution complete`);
    
    setIsRunning(false);
    toast.success('Code executed');
  };

  const clearOutput = () => {
    setOutput('');
    setPackagesInstalled(false);
    toast.info('Terminal cleared');
  };

  const copyCode = async () => {
    const code = currentTab === 'python' ? pythonCode : rCode;
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success('Copied to clipboard');
  };

  const currentEstimand = estimandsData.find(e => e.id === selectedEstimand);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Code Playground</h1>
          <p className="text-muted-foreground">
            Execute estimand examples with auto-installed packages
          </p>
        </div>

        {/* Estimand Selector */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">Select Estimand</label>
              <Select value={selectedEstimand} onValueChange={setSelectedEstimand}>
                <SelectTrigger className="w-full md:w-[400px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-[400px]">
                  {estimandsData.map(e => (
                    <SelectItem key={e.id} value={e.id}>
                      {e.short_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {currentEstimand && (
              <div className="flex flex-wrap gap-2">
                <Badge>{currentEstimand.tier}</Badge>
                <Badge variant="outline">{currentEstimand.framework}</Badge>
                <Badge variant="outline">{currentEstimand.design}</Badge>
              </div>
            )}
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Code Editor */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Code</h2>
              <div className="flex gap-2">
                {!packagesInstalled && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={installPackages}
                    disabled={isRunning}
                    className="gap-2"
                  >
                    <Package className="h-4 w-4" />
                    Install Packages
                  </Button>
                )}
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
                  disabled={isRunning || !packagesInstalled}
                  className="gap-2"
                >
                  <Play className="h-4 w-4" />
                  {isRunning ? 'Running...' : 'Run'}
                </Button>
              </div>
            </div>

            <Tabs value={currentTab} onValueChange={setCurrentTab}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="python">Python</TabsTrigger>
                <TabsTrigger value="r">R</TabsTrigger>
              </TabsList>
              
              <TabsContent value="python" className="mt-4">
                <textarea
                  value={pythonCode}
                  onChange={(e) => setPythonCode(e.target.value)}
                  placeholder="Python code will load here..."
                  className="w-full h-[500px] p-4 font-mono text-sm bg-terminal-bg text-terminal-fg rounded-lg border focus:outline-none focus:ring-2 focus:ring-accent resize-none"
                />
              </TabsContent>
              
              <TabsContent value="r" className="mt-4">
                <textarea
                  value={rCode}
                  onChange={(e) => setRCode(e.target.value)}
                  placeholder="R code will load here..."
                  className="w-full h-[500px] p-4 font-mono text-sm bg-terminal-bg text-terminal-fg rounded-lg border focus:outline-none focus:ring-2 focus:ring-accent resize-none"
                />
              </TabsContent>
            </Tabs>
          </div>

          {/* Output */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Console Output</h2>
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
                <div className="flex flex-col items-center justify-center h-full text-terminal-fg/50 text-center">
                  <Package className="h-12 w-12 mb-4 opacity-50" />
                  <p>Click "Install Packages" to begin</p>
                  <p className="text-xs mt-2">Then run your code</p>
                </div>
              )}
            </div>

            {/* Requirements Info */}
            <div className="p-4 rounded-lg border bg-muted/50">
              <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
                <Package className="h-4 w-4" />
                Auto-Requirements
              </h3>
              <div className="text-xs text-muted-foreground space-y-1">
                <p><strong>Python:</strong> numpy, scipy, scikit-learn, pandas</p>
                <p><strong>R:</strong> survival, glmnet, randomForest, ppcor</p>
                <p className="mt-2 pt-2 border-t">
                  Full execution requires Pyodide/WebR integration. This interface demonstrates structured playground with auto-package management.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TerminalView;