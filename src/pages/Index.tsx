import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Index = () => {
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
              <Link to="/learning">
                <Button size="lg" variant="secondary">
                  Get Started
                </Button>
              </Link>
              <Link to="/network">
                <Button size="lg" variant="outline" className="bg-white/10 hover:bg-white/20 text-white border-white/20">
                  Explore Network
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
};

export default Index;
