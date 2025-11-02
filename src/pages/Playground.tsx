import { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Play, RotateCcw, Download, BookOpen, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { GraphicsCanvas } from '@/components/GraphicsCanvas';
import { executePython, executeR } from '@/lib/codeExecution';
import { allLessons, lessonsByTier, type Lesson } from '@/data/lessons';
import { ScrollArea } from '@/components/ui/scroll-area';
import { translateCode, type Language } from '@/lib/codeTranslator';
import { ArrowRightLeft } from 'lucide-react';

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
  const [showLessons, setShowLessons] = useState(false);
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const [targetLanguage, setTargetLanguage] = useState<string>('r');
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
    terminal.writeln('Python & R Runtime with Package Installation');
    terminal.writeln('');
    terminal.writeln('Commands:');
    terminal.writeln('  Python: pip install <package>');
    terminal.writeln('  R: install.packages("<package>")');
    terminal.writeln('');
    terminal.write('$ ');

    let currentLine = '';

    terminal.onData(async (data) => {
      const code = data.charCodeAt(0);

      if (code === 13) { // Enter
        terminal.writeln('');
        
        if (currentLine.trim()) {
          const command = currentLine.trim();
          
          if (command.startsWith('pip install ')) {
            const pkg = command.substring(12).trim();
            terminal.writeln(`Installing Python package: ${pkg}...`);
            const { installPythonPackage } = await import('@/lib/codeExecution');
            const result = await installPythonPackage(pkg);
            terminal.writeln(result.error || result.output);
          } else if (command.includes('install.packages')) {
            const match = command.match(/install\.packages\(['"]([^'"]+)['"]\)/);
            if (match) {
              const pkg = match[1];
              terminal.writeln(`Installing R package: ${pkg}...`);
              const { installRPackage } = await import('@/lib/codeExecution');
              const result = await installRPackage(pkg);
              terminal.writeln(result.error || result.output);
            }
          } else {
            terminal.writeln(`Unknown command: ${command}`);
            terminal.writeln('Use "pip install <package>" or \'install.packages("<package>")\'');
          }
        }
        
        currentLine = '';
        terminal.write('$ ');
      } else if (code === 127) { // Backspace
        if (currentLine.length > 0) {
          currentLine = currentLine.slice(0, -1);
          terminal.write('\b \b');
        }
      } else if (code >= 32) { // Printable characters
        currentLine += data;
        terminal.write(data);
      }
    });

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
    if (!selectedLesson) {
      setCode(STARTER_CODE[value as keyof typeof STARTER_CODE] || '');
    } else {
      setCode(value === 'python' ? selectedLesson.pythonCode : selectedLesson.rCode);
    }
    setOutput('');
  };

  const handleLessonSelect = (lesson: Lesson) => {
    setSelectedLesson(lesson);
    // Auto-switch to Python (preferred) or R based on lesson content
    const targetLanguage = lesson.pythonCode ? 'python' : 'r';
    setLanguage(targetLanguage);
    setCode(targetLanguage === 'python' ? lesson.pythonCode : lesson.rCode);
    setOutput('');
    setShowLessons(false);
    toast({
      title: 'Lesson loaded',
      description: `${lesson.title} (${targetLanguage === 'python' ? 'Python' : 'R'})`,
    });
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
        
        // Write to terminal
        if (terminalInstance.current) {
          terminalInstance.current.writeln('=== HTML Rendered ===');
          terminalInstance.current.writeln('View the Preview tab to see output');
          terminalInstance.current.writeln('');
        }
      } else if (language === 'javascript') {
        try {
          const logs: string[] = [];
          const originalLog = console.log;
          console.log = (...args) => {
            logs.push(args.map(a => typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a)).join(' '));
          };

          // eslint-disable-next-line no-eval
          eval(code);

          console.log = originalLog;
          const outputText = logs.join('\n') || 'Code executed successfully (no output)';
          setOutput(outputText);
          
          // Write to terminal
          if (terminalInstance.current) {
            terminalInstance.current.writeln('=== JavaScript Output ===');
            terminalInstance.current.writeln(outputText);
            terminalInstance.current.writeln('');
          }
          
          setActiveView('output');
        } catch (error: any) {
          const errorMsg = `Error: ${error.message}`;
          setOutput(errorMsg);
          
          if (terminalInstance.current) {
            terminalInstance.current.writeln('=== JavaScript Error ===');
            terminalInstance.current.writeln(errorMsg);
            terminalInstance.current.writeln('');
          }
          
          setActiveView('output');
        }
      } else if (language === 'python') {
        const result = await executePython(code);
        const outputText = result.error || result.output;
        setOutput(outputText);
        
        // Write to terminal
        if (terminalInstance.current) {
          terminalInstance.current.writeln('=== Python Output ===');
          terminalInstance.current.writeln(outputText);
          terminalInstance.current.writeln('');
        }
        
        // Auto-switch to graphics tab if DAG code is detected
        if (code.includes('dag(') || code.includes('DAG(') || code.includes('create_dag')) {
          setActiveView('graphics');
        } else {
          setActiveView('output');
        }
      } else if (language === 'r') {
        const result = await executeR(code);
        const outputText = result.error || result.output;
        setOutput(outputText);
        
        // Write to terminal
        if (terminalInstance.current) {
          terminalInstance.current.writeln('=== R Output ===');
          terminalInstance.current.writeln(outputText);
          terminalInstance.current.writeln('');
        }
        
        // Auto-switch to graphics tab if DAG code is detected
        if (code.includes('dag(') || code.includes('DAG(') || code.includes('create_dag')) {
          setActiveView('graphics');
        } else {
          setActiveView('output');
        }
      }

      toast({
        title: 'Code executed',
        description: code.includes('dag(') || code.includes('DAG(') ? 'View graphics tab' : 
                     language === 'html' ? 'View preview tab' : 'Check the output tab',
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
      terminalInstance.current.writeln('Welcome to Causal Atlas IDE Terminal');
      terminalInstance.current.writeln('Python & R Runtime with Package Installation');
      terminalInstance.current.writeln('');
    }
    // Clear iframe preview
    if (iframeRef.current) {
      const doc = iframeRef.current.contentDocument;
      if (doc) {
        doc.open();
        doc.write('');
        doc.close();
      }
    }
    toast({
      title: 'Cleared',
      description: 'All outputs have been cleared',
    });
  };

  const translateCurrentCode = () => {
    try {
      const translated = translateCode(code, language as Language, targetLanguage as Language);
      setCode(translated);
      setLanguage(targetLanguage);
      toast({
        title: 'Code translated',
        description: `Translated from ${language} to ${targetLanguage}`,
      });
    } catch (error: any) {
      toast({
        title: 'Translation failed',
        description: error.message || 'Could not translate code',
        variant: 'destructive',
      });
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
          <Button 
            onClick={() => setShowLessons(!showLessons)} 
            variant={showLessons ? "default" : "outline"} 
            size="sm"
          >
            <BookOpen className="h-4 w-4 mr-2" />
            Lessons
          </Button>

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

          <div className="flex items-center gap-2">
            <ArrowRightLeft className="h-4 w-4 text-muted-foreground" />
            <Select value={targetLanguage} onValueChange={setTargetLanguage}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Translate to" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="python">Python</SelectItem>
                <SelectItem value="javascript">JavaScript</SelectItem>
                <SelectItem value="r">R</SelectItem>
                <SelectItem value="html">HTML</SelectItem>
              </SelectContent>
            </Select>
            <Button
              onClick={translateCurrentCode}
              variant="outline"
              size="sm"
              disabled={language === targetLanguage}
            >
              Translate
            </Button>
          </div>

          <Button onClick={runCode} disabled={running} size="sm">
            {running ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
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

        {selectedLesson && (
          <div className="text-sm text-muted-foreground">
            {selectedLesson.title}
          </div>
        )}
      </div>

      {/* Main Content */}
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Lessons Panel */}
        {showLessons && (
          <>
            <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
              <div className="h-full flex flex-col">
                <div className="border-b bg-muted px-4 py-2 text-sm font-medium">
                  Causal Inference Lessons
                </div>
                <ScrollArea className="flex-1">
                  <div className="p-4 space-y-4">
                    {Object.entries(lessonsByTier).map(([tier, lessons]) => (
                      <div key={tier}>
                        <h3 className="font-semibold text-sm mb-2">{tier}</h3>
                        <div className="space-y-1">
                          {lessons.map(lesson => (
                            <Button
                              key={lesson.id}
                              variant={selectedLesson?.id === lesson.id ? "secondary" : "ghost"}
                              className="w-full justify-start text-left h-auto py-2"
                              onClick={() => handleLessonSelect(lesson)}
                            >
                              <div className="flex flex-col items-start">
                                <div className="text-sm font-medium">{lesson.title}</div>
                                <div className="text-xs text-muted-foreground">{lesson.description.slice(0, 50)}...</div>
                              </div>
                            </Button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </ResizablePanel>
            <ResizableHandle />
          </>
        )}

        {/* Editor Panel */}
        <ResizablePanel defaultSize={showLessons ? 40 : 50} minSize={30}>
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
        <ResizablePanel defaultSize={showLessons ? 40 : 50} minSize={30}>
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
