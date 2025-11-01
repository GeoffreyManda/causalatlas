import Navigation from '@/components/Navigation';
import { useState } from 'react';
import { estimandsData } from '@/data/estimands';
import EstimandCard from '@/components/EstimandCard';
import { Badge } from '@/components/ui/badge';

const EstimandsLibrary = () => {
  const [selectedTier, setSelectedTier] = useState<string>('All');
  
  const tiers = ['All', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  
  const filteredEstimands = selectedTier === 'All' 
    ? estimandsData 
    : estimandsData.filter(e => e.tier === selectedTier);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-16">
        <div className="container">
          <h1 className="text-4xl font-bold text-primary-foreground mb-4">
            Estimand Library
          </h1>
          <p className="text-xl text-primary-foreground/90 max-w-3xl">
            Browse and explore causal estimands organized by complexity tier, from basic population effects to frontier methods
          </p>
        </div>
      </section>

      {/* Estimand Library */}
      <section className="py-12">
        <div className="container">
          <div className="mb-8">
            <h2 className="text-2xl font-bold mb-4">Filter by Difficulty Tier</h2>
            
            {/* Tier Filter */}
            <div className="flex flex-wrap gap-2">
              {tiers.map((tier) => (
                <Badge
                  key={tier}
                  variant={selectedTier === tier ? "default" : "outline"}
                  className="cursor-pointer px-4 py-2 text-sm hover:scale-105 transition-transform"
                  onClick={() => setSelectedTier(tier)}
                >
                  {tier}
                </Badge>
              ))}
            </div>
            
            <p className="text-sm text-muted-foreground mt-4">
              Showing {filteredEstimands.length} estimand{filteredEstimands.length !== 1 ? 's' : ''}
            </p>
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

export default EstimandsLibrary;
