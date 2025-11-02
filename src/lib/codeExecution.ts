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
    
    // Load common packages including matplotlib for plotting
    await pyodideInstance.loadPackage(['numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib']);
    console.log('Pyodide loaded with packages (numpy, scipy, scikit-learn, pandas, matplotlib)');
    
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

export async function installPythonPackage(packageName: string): Promise<{ output: string; error?: string }> {
  try {
    const pyodide = await initializePython();
    await pyodide.loadPackage(['micropip']);
    await pyodide.runPythonAsync(`
      import micropip
      await micropip.install('${packageName}')
    `);
    return { output: `Successfully installed ${packageName}` };
  } catch (error: any) {
    return { 
      output: '', 
      error: `Failed to install ${packageName}: ${error.message}`
    };
  }
}

export async function executePython(code: string): Promise<{ output: string; error?: string }> {
  try {
    const pyodide = await initializePython();
    
    // Check for pip install commands
    const pipMatch = code.match(/pip install ([a-zA-Z0-9_-]+)/);
    if (pipMatch) {
      const packageName = pipMatch[1];
      const installResult = await installPythonPackage(packageName);
      if (installResult.error) return installResult;
      // Remove the pip install line and continue
      code = code.replace(/pip install [a-zA-Z0-9_-]+\n?/, '');
    }
    
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

export async function installRPackage(packageName: string): Promise<{ output: string; error?: string }> {
  try {
    const webR = await initializeR();
    const shelter = await new webR.Shelter();
    
    try {
      await shelter.captureR(`install.packages('${packageName}', repos='https://cran.r-project.org')`, {
        withAutoprint: false,
        captureStreams: true,
        captureConditions: false,
      });
      return { output: `Successfully installed ${packageName}` };
    } finally {
      shelter.purge();
    }
  } catch (error: any) {
    return { 
      output: '', 
      error: `Failed to install ${packageName}: ${error.message}`
    };
  }
}

export async function executeR(code: string): Promise<{ output: string; error?: string }> {
  try {
    const webR = await initializeR();
    
    // Check for install.packages commands
    const installMatch = code.match(/install\.packages\(['"]([^'"]+)['"]\)/);
    if (installMatch) {
      const packageName = installMatch[1];
      const installResult = await installRPackage(packageName);
      if (installResult.error) return installResult;
      // Remove the install.packages line and continue
      code = code.replace(/install\.packages\(['"][^'"]+['"]\)\n?/, '');
    }
    
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
