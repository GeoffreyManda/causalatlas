import Navigation from '@/components/Navigation';
import { useState } from 'react';
import { estimandsData } from '@/data/estimands';
import EstimandCard from '@/components/EstimandCard';
import { Badge } from '@/components/ui/badge';

const EstimandsLibrary = () => {
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [selectedDesign, setSelectedDesign] = useState<string>('all');
  
  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(estimandsData.map(e => e.framework)))];
  const designs = ['all', ...Array.from(new Set(estimandsData.map(e => e.design))).sort()];
  
  const filteredEstimands = estimandsData.filter(e => {
    if (selectedTier !== 'all' && e.tier !== selectedTier) return false;
    if (selectedFramework !== 'all' && e.framework !== selectedFramework) return false;
    if (selectedDesign !== 'all' && e.design !== selectedDesign) return false;
    return true;
  });

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
          {/* Filters */}
          <div className="mb-8 p-6 rounded-lg border bg-card">
            <h2 className="text-2xl font-bold mb-6">Filters</h2>
            
            <div className="grid md:grid-cols-3 gap-6">
              {/* Tier Filter */}
              <div>
                <h3 className="text-sm font-medium mb-3">Filter by Tier</h3>
                <div className="flex flex-wrap gap-2">
                  {tiers.map((tier) => (
                    <Badge
                      key={tier}
                      variant={selectedTier === tier ? "default" : "outline"}
                      className="cursor-pointer px-4 py-2 text-sm hover:scale-105 transition-transform"
                      onClick={() => setSelectedTier(tier)}
                    >
                      {tier === 'all' ? 'All Tiers' : tier}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Framework Filter */}
              <div>
                <h3 className="text-sm font-medium mb-3">Filter by Framework</h3>
                <div className="flex flex-wrap gap-2">
                  {frameworks.map((fw) => (
                    <Badge
                      key={fw}
                      variant={selectedFramework === fw ? "default" : "outline"}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedFramework(fw)}
                    >
                      {fw === 'all' ? 'All Frameworks' : fw.replace(/([A-Z])/g, ' $1').trim()}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Design Filter */}
              <div>
                <h3 className="text-sm font-medium mb-3">Filter by Study Design</h3>
                <div className="flex flex-wrap gap-2">
                  {designs.map((design) => (
                    <Badge
                      key={design}
                      variant={selectedDesign === design ? "default" : "outline"}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedDesign(design)}
                    >
                      {design === 'all' ? 'All Designs' : design.replace(/_/g, ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
            
            <p className="text-sm text-muted-foreground mt-4">
              Showing {filteredEstimands.length} of {estimandsData.length} estimand{filteredEstimands.length !== 1 ? 's' : ''}
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
