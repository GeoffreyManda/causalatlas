import Navigation from '@/components/Navigation';
import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { estimandsData } from '@/data/estimands';
import EstimandCard from '@/components/EstimandCard';
import { Badge } from '@/components/ui/badge';

const EstimandsLibrary = () => {
  const [searchParams] = useSearchParams();
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [selectedDesign, setSelectedDesign] = useState<string>('all');
  const [selectedFamily, setSelectedFamily] = useState<string>('all');
  
  // Apply filters from URL params on mount
  useEffect(() => {
    const tier = searchParams.get('tier');
    const framework = searchParams.get('framework');
    const design = searchParams.get('design');
    const family = searchParams.get('family');
    
    if (tier) setSelectedTier(tier);
    if (framework) setSelectedFramework(framework);
    if (design) setSelectedDesign(design);
    if (family) setSelectedFamily(family);
  }, [searchParams]);
  
  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(estimandsData.map(e => e.framework)))];
  const designs = ['all', ...Array.from(new Set(estimandsData.map(e => e.design))).sort()];
  const families = ['all', ...Array.from(new Set(estimandsData.map(e => e.estimand_family))).sort()];
  
  const filteredEstimands = estimandsData.filter(e => {
    if (selectedTier !== 'all' && e.tier !== selectedTier) return false;
    if (selectedFramework !== 'all' && e.framework !== selectedFramework) return false;
    if (selectedDesign !== 'all' && e.design !== selectedDesign) return false;
    if (selectedFamily !== 'all' && e.estimand_family !== selectedFamily) return false;
    return true;
  }).sort((a, b) => {
    // Define tier order
    const tierOrder = { 'Basic': 1, 'Intermediate': 2, 'Advanced': 3, 'Frontier': 4 };
    const tierA = tierOrder[a.tier as keyof typeof tierOrder] || 5;
    const tierB = tierOrder[b.tier as keyof typeof tierOrder] || 5;
    
    if (tierA !== tierB) return tierA - tierB;
    
    // Within each tier, define logical progression order
    const logicalOrder: { [key: string]: number } = {
      // Basic: Simple population effects first
      'ATE': 1,
      'ATT': 2,
      'ATC': 3,
      'CATE': 4,
      'ITE': 5,
      
      // Intermediate: Standard causal methods
      'NDE': 10,
      'NIE': 11,
      'IV': 12,
      'LATE': 13,
      'RD': 14,
      'DID': 15,
      
      // Advanced: Complex/dynamic methods
      'TIME_VARYING': 20,
      'DYNAMIC': 21,
      'MEDIATION_COMPLEX': 22,
      'QTE': 23,
      'SURVIVAL': 24,
      
      // Frontier: ML and modern methods
      'ML_CATE': 30,
      'DEEP_LEARNING': 31,
      'ADAPTIVE': 32,
    };
    
    const orderA = logicalOrder[a.id] || 999;
    const orderB = logicalOrder[b.id] || 999;
    
    if (orderA !== orderB) return orderA - orderB;
    
    // If not in logical order map, sort by family then alphabetically
    if (a.estimand_family !== b.estimand_family) {
      return a.estimand_family.localeCompare(b.estimand_family);
    }
    
    return a.short_name.localeCompare(b.short_name);
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
            
            <div className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
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

                {/* Family Filter */}
                <div>
                  <h3 className="text-sm font-medium mb-3">Filter by Type</h3>
                  <div className="flex flex-wrap gap-2">
                    {families.map((family) => (
                      <Badge
                        key={family}
                        variant={selectedFamily === family ? "default" : "outline"}
                        className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                        onClick={() => setSelectedFamily(family)}
                      >
                        {family === 'all' ? 'All Types' : family === 'SurvivalTimeToEvent' ? 'Survival/Time-to-Event' : family.replace(/([A-Z])/g, ' $1').trim()}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
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
