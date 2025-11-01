import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';
import { useState } from 'react';
import { estimandsData } from '@/data/estimands';
import EstimandCard from '@/components/EstimandCard';
import { Badge } from '@/components/ui/badge';

const Index = () => {
  const [selectedTier, setSelectedTier] = useState<string>('All');
  
  const tiers = ['All', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  
  const filteredEstimands = selectedTier === 'All' 
    ? estimandsData 
    : estimandsData.filter(e => e.tier === selectedTier);

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

      {/* Estimand Library */}
      <section className="py-12">
        <div className="container">
          <div className="mb-8">
            <h2 className="text-3xl font-bold mb-4">Estimand Library</h2>
            <p className="text-muted-foreground mb-6">
              Browse and explore causal estimands organized by complexity tier
            </p>
            
            {/* Tier Filter */}
            <div className="flex flex-wrap gap-2">
              {tiers.map((tier) => (
                <Badge
                  key={tier}
                  variant={selectedTier === tier ? "default" : "outline"}
                  className="cursor-pointer px-4 py-2 text-sm"
                  onClick={() => setSelectedTier(tier)}
                >
                  {tier}
                </Badge>
              ))}
            </div>
          </div>
          
          {/* Estimand Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredEstimands.map((estimand) => (
              <EstimandCard key={estimand.id} estimand={estimand} />
            ))}
          </div>
          
          {filteredEstimands.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              No estimands found for this tier
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default Index;
