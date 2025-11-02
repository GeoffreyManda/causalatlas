import { loadPyodide } from 'pyodide';
import { WebR } from 'webr';

let pyodideInstance: any = null;
let webRInstance: WebR | null = null;
let pyodideLoading = false;
let webRLoading = false;

export async function initializePython() {
  if (pyodideInstance) return pyodideInstance;
  if (pyodideLoading) {
    // Wait for loading to complete
    while (pyodideLoading) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return pyodideInstance;
  }

  pyodideLoading = true;
  try {
    console.log('Loading Pyodide...');
    pyodideInstance = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
    });
    
    // Load common packages
    await pyodideInstance.loadPackage(['numpy', 'scipy', 'scikit-learn', 'pandas']);
    console.log('Pyodide loaded with packages');
    
    return pyodideInstance;
  } finally {
    pyodideLoading = false;
  }
}

export async function initializeR() {
  if (webRInstance) return webRInstance;
  if (webRLoading) {
    while (webRLoading) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return webRInstance;
  }

  webRLoading = true;
  try {
    console.log('Loading WebR...');
    webRInstance = new WebR();
    await webRInstance.init();
    console.log('WebR loaded');
    
    return webRInstance;
  } finally {
    webRLoading = false;
  }
}

export async function executePython(code: string): Promise<{ output: string; error?: string }> {
  try {
    const pyodide = await initializePython();
    
    // Capture stdout
    let output = '';
    pyodide.setStdout({
      batched: (text: string) => {
        output += text + '\n';
      }
    });

    // Execute code
    await pyodide.runPythonAsync(code);
    
    return { output: output || 'Code executed successfully' };
  } catch (error: any) {
    return { 
      output: '', 
      error: `Python Error: ${error.message}`
    };
  }
}

export async function executeR(code: string): Promise<{ output: string; error?: string }> {
  try {
    const webR = await initializeR();
    
    let output = '';
    
    // Capture output
    const shelter = await new webR.Shelter();
    try {
      const result = await shelter.captureR(code, {
        withAutoprint: true,
        captureStreams: true,
        captureConditions: false,
      });
      
      const outputMessages = result.output.filter(
        (msg: any) => msg.type === 'stdout' || msg.type === 'stderr'
      );
      
      output = outputMessages.map((msg: any) => msg.data).join('\n');
      
      return { output: output || 'Code executed successfully' };
    } finally {
      shelter.purge();
    }
  } catch (error: any) {
    return { 
      output: '', 
      error: `R Error: ${error.message}`
    };
  }
}
