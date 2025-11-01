import { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Play, RotateCcw, Download, Upload, Settings } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { GraphicsCanvas } from '@/components/GraphicsCanvas';

const STARTER_CODE = {
  python: `# Python Code Editor with Visualization
import math
import random

# Basic output
print("Hello from Python!")
print(f"Random number: {random.randint(1, 100)}")
print(f"Pi value: {math.pi}")

# Generate sample data for plotting
x = list(range(20))
y = [random.random() * 100 + math.sin(i/3) * 50 for i in x]

print("\\nData points generated:")
print(f"X: {x[:5]}...")
print(f"Y: {y[:5]}...")

# Simulate plotting
plot(x, y)
print("\\nPlot created!")
`,
  javascript: `// JavaScript with D3 Visualization
console.log("Hello from JavaScript!");

// Generate sample data
const data = Array.from({length: 20}, (_, i) => ({
  x: i,
  y: Math.random() * 100 + Math.sin(i / 3) * 50
}));

console.log("Data generated:", data.slice(0, 3));
console.log("Data points:", data.length);

// Trigger visualization
plot(data);
console.log("Visualization rendered!");
`,
  r: `# R Statistical Computing
print("Hello from R!")

# Generate random data
x <- 1:20
y <- runif(20, 0, 100) + sin(x/3) * 50

# Display summary
print(summary(y))
cat("\\nMean:", mean(y), "\\n")
cat("Std Dev:", sd(y), "\\n")

# Create plot
plot(x, y, type="l", col="blue", 
     main="Sample R Plot",
     xlab="X", ylab="Y")
`,
  html: `<!DOCTYPE html>
<html>
<head>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', Arial, sans-serif; 
      padding: 40px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .container {
      background: white;
      padding: 40px;
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      max-width: 600px;
    }
    h1 { 
      color: #667eea; 
      margin-bottom: 20px;
      font-size: 32px;
    }
    .box { 
      width: 100%; 
      height: 200px; 
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 15px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 24px;
      margin-top: 20px;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.05); }
    }
    p { 
      color: #666; 
      line-height: 1.6; 
      margin-top: 15px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>HTML Playground</h1>
    <p>Edit this HTML to create amazing things!</p>
    <div class="box">Hello World! 👋</div>
    <p>Try modifying the colors, text, or animations above.</p>
  </div>
</body>
</html>`
};

const Playground = () => {
  const { toast } = useToast();
  const [code, setCode] = useState(STARTER_CODE.python);
  const [language, setLanguage] = useState('python');
  const [output, setOutput] = useState('');
  const [running, setRunning] = useState(false);
  const [activeView, setActiveView] = useState('output');
  const terminalRef = useRef<HTMLDivElement>(null);
  const terminalInstance = useRef<Terminal | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Initialize terminal
  useEffect(() => {
    if (!terminalRef.current) return;

    const terminal = new Terminal({
      cursorBlink: true,
      fontSize: 14,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: '#1e1e1e',
        foreground: '#d4d4d4',
      },
    });

    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(terminalRef.current);
    fitAddon.fit();

    terminal.writeln('Welcome to Causal Atlas IDE Terminal');
    terminal.writeln('Type your commands here...');
    terminal.writeln('');

    terminalInstance.current = terminal;

    const handleResize = () => fitAddon.fit();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      terminal.dispose();
    };
  }, []);

  const handleLanguageChange = (value: string) => {
    setLanguage(value);
    setCode(STARTER_CODE[value as keyof typeof STARTER_CODE] || '');
    setOutput('');
  };

  const runCode = async () => {
    if (!code.trim()) {
      toast({
        title: 'No code to run',
        description: 'Please write some code first',
        variant: 'destructive',
      });
      return;
    }

    setRunning(true);
    setOutput('Running...\n');

    try {
      if (language === 'html') {
        // Render HTML directly in iframe
        if (iframeRef.current) {
          const doc = iframeRef.current.contentDocument;
          if (doc) {
            doc.open();
            doc.write(code);
            doc.close();
          }
        }
        setActiveView('preview');
        setOutput('HTML rendered in preview tab');
      } else if (language === 'javascript') {
        // Execute JavaScript locally
        try {
          const logs: string[] = [];
          const originalLog = console.log;
          console.log = (...args) => {
            logs.push(args.map(a => typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a)).join(' '));
          };

          // eslint-disable-next-line no-eval
          eval(code);

          console.log = originalLog;
          setOutput(logs.join('\n') || 'Code executed successfully (no output)');
        } catch (error: any) {
          setOutput(`Error: ${error.message}`);
        }
      } else {
        // For Python and R, call edge function
        const { data, error } = await supabase.functions.invoke('execute-code', {
          body: { code, language },
        });

        if (error) throw error;

        setOutput(data.output || data.error || 'No output');
        
        if (terminalInstance.current && data.output) {
          terminalInstance.current.writeln(data.output);
        }
      }

      toast({
        title: 'Code executed',
        description: 'Check the output below',
      });
    } catch (error: any) {
      const errorMsg = error.message || 'Failed to execute code';
      setOutput(`Error: ${errorMsg}`);
      toast({
        title: 'Execution failed',
        description: errorMsg,
        variant: 'destructive',
      });
    } finally {
      setRunning(false);
    }
  };

  const clearOutput = () => {
    setOutput('');
    if (terminalInstance.current) {
      terminalInstance.current.clear();
    }
  };

  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `code.${language === 'python' ? 'py' : language === 'javascript' ? 'js' : language === 'r' ? 'r' : 'html'}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-screen bg-background flex flex-col pt-16">
      {/* Toolbar */}
      <div className="border-b bg-card px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Select value={language} onValueChange={handleLanguageChange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select language" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="python">Python</SelectItem>
              <SelectItem value="javascript">JavaScript</SelectItem>
              <SelectItem value="r">R</SelectItem>
              <SelectItem value="html">HTML</SelectItem>
            </SelectContent>
          </Select>

          <Button onClick={runCode} disabled={running} size="sm">
            <Play className="h-4 w-4 mr-2" />
            Run
          </Button>

          <Button onClick={clearOutput} variant="outline" size="sm">
            <RotateCcw className="h-4 w-4 mr-2" />
            Clear
          </Button>

          <Button onClick={downloadCode} variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>

        <Button variant="ghost" size="sm">
          <Settings className="h-4 w-4 mr-2" />
          Settings
        </Button>
      </div>

      {/* Main Content */}
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Editor Panel */}
        <ResizablePanel defaultSize={50} minSize={30}>
          <div className="h-full flex flex-col">
            <div className="border-b bg-muted px-4 py-2 text-sm font-medium">
              Editor
            </div>
            <div className="flex-1">
              <Editor
                height="100%"
                language={language === 'r' ? 'r' : language}
                value={code}
                onChange={(value) => setCode(value || '')}
                theme="vs-dark"
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: 'on',
                  roundedSelection: false,
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                }}
              />
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle />

        {/* Output Panel */}
        <ResizablePanel defaultSize={50} minSize={30}>
          <div className="h-full flex flex-col">
            <Tabs value={activeView} onValueChange={setActiveView} className="flex-1 flex flex-col">
              <div className="border-b bg-muted">
                <TabsList className="h-10 bg-transparent px-2">
                  <TabsTrigger value="output">Console Output</TabsTrigger>
                  <TabsTrigger value="terminal">Terminal</TabsTrigger>
                  <TabsTrigger value="preview">Preview</TabsTrigger>
                  <TabsTrigger value="graphics">Graphics</TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="output" className="flex-1 m-0">
                <div className="h-full bg-black text-green-400 p-4 font-mono text-sm overflow-auto">
                  <pre className="whitespace-pre-wrap">{output || '// Output will appear here...'}</pre>
                </div>
              </TabsContent>

              <TabsContent value="terminal" className="flex-1 m-0">
                <div ref={terminalRef} className="h-full" />
              </TabsContent>

              <TabsContent value="preview" className="flex-1 m-0">
                <iframe
                  ref={iframeRef}
                  className="w-full h-full border-0 bg-white"
                  title="Preview"
                  sandbox="allow-scripts"
                />
              </TabsContent>

              <TabsContent value="graphics" className="flex-1 m-0">
                <GraphicsCanvas code={code} language={language} />
              </TabsContent>
            </Tabs>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
};

export default Playground;
