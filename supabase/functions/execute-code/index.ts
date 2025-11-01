import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { code, language } = await req.json();

    if (!code || !language) {
      throw new Error('Code and language are required');
    }

    console.log(`Executing ${language} code:`, code.substring(0, 100));

    let output = '';
    let error = '';

    if (language === 'python') {
      // Simulate Python execution (in real implementation, use a sandboxed Python runtime)
      output = `Python execution simulation:\n${code}\n\nNote: This is a simulated output. For real Python execution, integrate with a Python runtime service.`;
    } else if (language === 'r') {
      // Simulate R execution
      output = `R execution simulation:\n${code}\n\nNote: This is a simulated output. For real R execution, integrate with an R runtime service.`;
    } else {
      error = `Language ${language} is not supported on the server side. Use JavaScript for client-side execution.`;
    }

    return new Response(
      JSON.stringify({ output, error }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (err) {
    console.error('Execution error:', err);
    return new Response(
      JSON.stringify({ 
        error: err instanceof Error ? err.message : 'Unknown error occurred' 
      }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});
