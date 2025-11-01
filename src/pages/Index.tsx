import Navigation from '@/components/Navigation';
import { Book, Network as NetworkIcon, Terminal as TerminalIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link, useNavigate } from 'react-router-dom';

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-20">
        <div className="container">
          <div className="max-w-3xl">
            <h1 className="text-5xl font-bold text-primary-foreground mb-4">
              Causal Estimand Atlas
            </h1>
            <p className="text-xl text-primary-foreground/90 mb-6">
              A comprehensive, publication-grade guide to causal inference estimands organized by Framework → Design → Family, 
              with interactive tutorials and executable code examples.
            </p>
            <div className="flex gap-4">
              <Link to="/network">
                <Button size="lg" variant="secondary" className="gap-2">
                  <NetworkIcon className="h-5 w-5" />
                  Explore Network
                </Button>
              </Link>
              <Link to="/playground">
                <Button size="lg" variant="outline" className="gap-2 bg-white/10 hover:bg-white/20 text-white border-white/20">
                  <TerminalIcon className="h-5 w-5" />
                  Open Playground
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-12 border-b">
        <div className="container">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          <div 
            onClick={() => navigate('/learning')}
            className="flex flex-col items-center text-center p-4 md:p-6 rounded-lg bg-gradient-card cursor-pointer hover:scale-105 transition-transform"
          >
            <div className="h-12 w-12 rounded-lg bg-tier-basic flex items-center justify-center mb-4">
              <Book className="h-6 w-6 text-white" />
            </div>
            <h3 className="font-semibold text-base md:text-lg mb-2">Structured Learning</h3>
            <p className="text-xs md:text-sm text-muted-foreground">
              Choose between Causal Theory and Causal Estimands
            </p>
          </div>
          <div 
            onClick={() => navigate('/network')}
            className="flex flex-col items-center text-center p-4 md:p-6 rounded-lg bg-gradient-card cursor-pointer hover:scale-105 transition-transform"
          >
            <div className="h-12 w-12 rounded-lg bg-tier-intermediate flex items-center justify-center mb-4">
              <NetworkIcon className="h-6 w-6 text-white" />
            </div>
            <h3 className="font-semibold text-base md:text-lg mb-2">Interactive Network</h3>
            <p className="text-xs md:text-sm text-muted-foreground">
              Visualize relationships between frameworks and estimands
            </p>
          </div>
          <div 
            onClick={() => navigate('/playground')}
            className="flex flex-col items-center text-center p-4 md:p-6 rounded-lg bg-gradient-card cursor-pointer hover:scale-105 transition-transform"
          >
            <div className="h-12 w-12 rounded-lg bg-tier-advanced flex items-center justify-center mb-4">
              <TerminalIcon className="h-6 w-6 text-white" />
            </div>
            <h3 className="font-semibold text-base md:text-lg mb-2">Executable Playground</h3>
            <p className="text-xs md:text-sm text-muted-foreground">
              Run Python examples in your browser
            </p>
          </div>
        </div>
        </div>
      </section>
    </div>
  );
};

export default Index;
