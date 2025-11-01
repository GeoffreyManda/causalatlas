import { useState, useEffect } from 'react';
import { useSearchParams, useLocation, useNavigate } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Play, Trash2, Copy, Check, Package, Download, ArrowLeft } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { estimandsData } from '@/data/estimands';
import { getRequirements } from '@/data/requirements';

const TerminalView = () => {
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedEstimand, setSelectedEstimand] = useState<string>('ate');
  const [pythonCode, setPythonCode] = useState('');
  const [rCode, setRCode] = useState('');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [currentTab, setCurrentTab] = useState('python');
  const [copied, setCopied] = useState(false);
  const [packagesInstalled, setPackagesInstalled] = useState(false);
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [pyodideInstance, setPyodideInstance] = useState<any>(null);

  // Get referrer from state or default to playground
  const referrer = (location.state as any)?.from || '/estimands';

  // Filter estimands
  const filteredEstimands = estimandsData.filter(e => {
    if (selectedTier !== 'all' && e.tier !== selectedTier) return false;
    if (selectedFramework !== 'all' && e.framework !== selectedFramework) return false;
    return true;
  });

  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(estimandsData.map(e => e.framework)))];

  // Ensure Pyodide is loaded only once
  const ensurePyodide = async () => {
    if (pyodideInstance) return pyodideInstance;
    // Load script if not already present
    if (!(window as any).loadPyodide) {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load Pyodide'));
        document.head.appendChild(script);
      });
    }
    const pyodide = await (window as any).loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/' });
    setPyodideInstance(pyodide);
    return pyodide;
  };

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
    const lang = currentTab;
    if (lang === 'r') {
      toast.warning('R execution will be enabled with WebR shortly. Python is fully executable now.');
      return;
    }
    setIsRunning(true);
    setOutput('📦 Resolving Python packages via Pyodide...\n\n');
    try {
      const pyodide = await ensurePyodide();
      const req = getRequirements(selectedEstimand).python;
      // Always ensure micropip and core scientific stack
      const toLoad = Array.from(new Set(['micropip', ...req]));
      for (const pkg of toLoad) {
        setOutput(prev => prev + `→ Loading ${pkg}...\n`);
        await pyodide.loadPackage(pkg).catch(async () => {
          // Try pip install pure-python wheels if not a pyodide pkg
          if (pkg !== 'micropip') {
            await pyodide.runPythonAsync(`import micropip; await micropip.install('${pkg}')`);
          }
        });
        setOutput(prev => prev + `✓ ${pkg} loaded\n`);
      }
      setOutput(prev => prev + '\n✅ Packages ready.');
      setPackagesInstalled(true);
    } catch (e: any) {
      console.error(e);
      setOutput(prev => prev + `\n❌ Package setup failed: ${e.message || e}`);
      setPackagesInstalled(false);
    } finally {
      setIsRunning(false);
    }
  };

  const runCode = async () => {
    if (!packagesInstalled) {
      toast.error('Please install packages first');
      return;
    }

    const code = currentTab === 'python' ? pythonCode : rCode;
    if (!code.trim()) {
      toast.error('No code to run');
      return;
    }

    setIsRunning(true);
    setOutput(prev => prev + `\n\n=== Running ${currentTab.toUpperCase()} ===\n\n`);

    try {
      if (currentTab === 'python') {
        const pyodide = await ensurePyodide();
        // Redirect stdout/stderr to buffer and exec provided code via JS bridge
        pyodide.globals.set('PY_CODE', code);
        await pyodide.runPythonAsync(`
import sys, io
_buf = io.StringIO()
_stdout, _stderr = sys.stdout, sys.stderr
sys.stdout = _buf
sys.stderr = _buf
try:
  exec(PY_CODE, globals())
except Exception as e:
  import traceback
  traceback.print_exc()
finally:
  sys.stdout = _stdout
  sys.stderr = _stderr
OUT_VAL = _buf.getvalue()
`);
        const out = pyodide.globals.get('OUT_VAL') as string;
        setOutput(prev => prev + (out || '(no output)'));
      } else {
        toast.warning('R execution via WebR is being enabled. For now, use Python.');
      }
    } catch (e: any) {
      console.error(e);
      setOutput(prev => prev + `\n❌ Runtime error: ${e.message || e}`);
    } finally {
      setIsRunning(false);
    }
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

  const downloadCode = () => {
    const code = currentTab === 'python' ? pythonCode : rCode;
    const filename = `${selectedEstimand}_${currentTab}.${currentTab === 'python' ? 'py' : 'R'}`;
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Downloaded ${filename}`);
  };

  const currentEstimand = estimandsData.find(e => e.id === selectedEstimand);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        {/* Back button */}
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            onClick={() => navigate(referrer)}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <h1 className="text-3xl font-bold">Code Playground</h1>
          <div className="w-24" /> {/* Spacer for centering */}
        </div>

        {/* Filters */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Filter by Tier</label>
              <div className="flex flex-wrap gap-2">
                {tiers.map(tier => (
                  <Badge
                    key={tier}
                    variant={selectedTier === tier ? 'default' : 'outline'}
                    className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                    onClick={() => setSelectedTier(tier)}
                  >
                    {tier === 'all' ? 'All' : tier}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Filter by Framework</label>
              <div className="flex flex-wrap gap-2">
                {frameworks.map(fw => (
                  <Badge
                    key={fw}
                    variant={selectedFramework === fw ? 'default' : 'outline'}
                    className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                    onClick={() => setSelectedFramework(fw)}
                  >
                    {fw === 'all' ? 'All' : fw.replace(/([A-Z])/g, ' $1').trim()}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Showing {filteredEstimands.length} of {estimandsData.length} estimands
          </p>
        </div>

        {/* Estimand Selector */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex-1 w-full">
              <label className="text-sm font-medium mb-2 block">Select Estimand</label>
              <Select value={selectedEstimand} onValueChange={setSelectedEstimand}>
                <SelectTrigger className="w-full bg-card">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-[400px]">
                  {filteredEstimands.map(e => (
                    <SelectItem key={e.id} value={e.id}>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">{e.tier}</Badge>
                        {e.short_name}
                      </div>
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
                  onClick={downloadCode}
                  disabled={!pythonCode && !rCode}
                  className="gap-2"
                >
                  <Download className="h-4 w-4" />
                  Download
                </Button>
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
                Requirements for {currentEstimand?.short_name}
              </h3>
              <div className="text-xs text-muted-foreground space-y-1">
                <p><strong>Python:</strong> {getRequirements(selectedEstimand).python.join(', ')}</p>
                <p><strong>R:</strong> {getRequirements(selectedEstimand).r.join(', ')}</p>
                <p className="mt-2 pt-2 border-t">
                  Python runs locally via Pyodide (real execution). R via WebR is being enabled next.
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