import { useState } from 'react';
import Navigation from '@/components/Navigation';
import EstimandCard from '@/components/EstimandCard';
import { estimandsData, Estimand } from '@/data/estimands';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Search, Book, Network as NetworkIcon, Terminal as TerminalIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Index = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [tierFilter, setTierFilter] = useState<string>('all');
  const [frameworkFilter, setFrameworkFilter] = useState<string>('all');
  const [selectedEstimand, setSelectedEstimand] = useState<Estimand | null>(null);

  const filteredEstimands = estimandsData.filter((estimand) => {
    const matchesSearch = estimand.short_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         estimand.estimand_family.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesTier = tierFilter === 'all' || estimand.tier === tierFilter;
    const matchesFramework = frameworkFilter === 'all' || estimand.framework === frameworkFilter;
    return matchesSearch && matchesTier && matchesFramework;
  });

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
              <Link to="/terminal">
                <Button size="lg" variant="outline" className="gap-2 bg-white/10 hover:bg-white/20 text-white border-white/20">
                  <TerminalIcon className="h-5 w-5" />
                  Try Terminal
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
          <div className="flex flex-col items-center text-center p-4 md:p-6 rounded-lg bg-gradient-card">
            <div className="h-12 w-12 rounded-lg bg-tier-basic flex items-center justify-center mb-4">
              <Book className="h-6 w-6 text-white" />
            </div>
            <h3 className="font-semibold text-base md:text-lg mb-2">Structured Learning</h3>
            <p className="text-xs md:text-sm text-muted-foreground">
              From basic RCT concepts to frontier deep learning methods
            </p>
          </div>
          <div className="flex flex-col items-center text-center p-4 md:p-6 rounded-lg bg-gradient-card">
            <div className="h-12 w-12 rounded-lg bg-tier-intermediate flex items-center justify-center mb-4">
              <NetworkIcon className="h-6 w-6 text-white" />
            </div>
            <h3 className="font-semibold text-base md:text-lg mb-2">Interactive Network</h3>
            <p className="text-xs md:text-sm text-muted-foreground">
              Visualize relationships between frameworks and estimands
            </p>
          </div>
          <div className="flex flex-col items-center text-center p-4 md:p-6 rounded-lg bg-gradient-card">
            <div className="h-12 w-12 rounded-lg bg-tier-advanced flex items-center justify-center mb-4">
              <TerminalIcon className="h-6 w-6 text-white" />
            </div>
            <h3 className="font-semibold text-base md:text-lg mb-2">Executable Code</h3>
            <p className="text-xs md:text-sm text-muted-foreground">
              Run Python and R examples in your browser
            </p>
          </div>
        </div>
        </div>
      </section>

      {/* Estimands Library */}
      <section className="py-12">
        <div className="container">
          <div className="mb-8">
            <h2 className="text-3xl font-bold mb-2">Estimand Library</h2>
            <p className="text-muted-foreground">Browse and filter through our comprehensive collection</p>
          </div>

          {/* Filters */}
          <div className="flex flex-col md:flex-row gap-4 mb-8">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search estimands..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={tierFilter} onValueChange={setTierFilter}>
              <SelectTrigger className="w-full md:w-[180px]">
                <SelectValue placeholder="Tier" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Tiers</SelectItem>
                <SelectItem value="Basic">Basic</SelectItem>
                <SelectItem value="Intermediate">Intermediate</SelectItem>
                <SelectItem value="Advanced">Advanced</SelectItem>
                <SelectItem value="Frontier">Frontier</SelectItem>
              </SelectContent>
            </Select>
            <Select value={frameworkFilter} onValueChange={setFrameworkFilter}>
              <SelectTrigger className="w-full md:w-[220px]">
                <SelectValue placeholder="Framework" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Frameworks</SelectItem>
                <SelectItem value="PotentialOutcomes">Potential Outcomes</SelectItem>
                <SelectItem value="SCM">SCM</SelectItem>
                <SelectItem value="PrincipalStratification">Principal Stratification</SelectItem>
                <SelectItem value="ProximalNegativeControl">Proximal/Negative Control</SelectItem>
                <SelectItem value="BayesianDecision">Bayesian Decision</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Results */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredEstimands.map((estimand) => (
              <EstimandCard
                key={estimand.id}
                estimand={estimand}
                onClick={() => setSelectedEstimand(estimand)}
              />
            ))}
          </div>

          {filteredEstimands.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No estimands found matching your filters</p>
            </div>
          )}
        </div>
      </section>

      {/* Estimand Detail Dialog */}
      <Dialog open={!!selectedEstimand} onOpenChange={() => setSelectedEstimand(null)}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          {selectedEstimand && (
            <>
              <DialogHeader>
                <DialogTitle className="text-2xl">{selectedEstimand.short_name}</DialogTitle>
                <DialogDescription className="flex flex-wrap gap-2 mt-2">
                  <Badge className="bg-tier-basic text-white">{selectedEstimand.tier}</Badge>
                  <Badge variant="outline">{selectedEstimand.framework}</Badge>
                  <Badge variant="outline">{selectedEstimand.design}</Badge>
                  <Badge variant="outline">{selectedEstimand.estimand_family}</Badge>
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-6 mt-4">
                <div>
                  <h4 className="font-semibold mb-2">Definition</h4>
                  <code className="block bg-muted p-3 rounded text-sm">{selectedEstimand.definition_tex}</code>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Identification Formula</h4>
                  <code className="block bg-muted p-3 rounded text-sm">{selectedEstimand.identification_formula_tex}</code>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Assumptions</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {selectedEstimand.assumptions.map((assumption, idx) => (
                      <li key={idx} className="text-sm text-muted-foreground">{assumption}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Estimators</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedEstimand.estimators.map((estimator, idx) => (
                      <Badge key={idx} variant="secondary">{estimator}</Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Properties</h4>
                  <div className="flex gap-2">
                    <Badge>{selectedEstimand.discovery_status}</Badge>
                    <Badge>EIF: {selectedEstimand.eif_status}</Badge>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">References</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {selectedEstimand.references.map((ref, idx) => (
                      <li key={idx} className="text-sm text-muted-foreground">
                        {ref.authors} ({ref.year}). {ref.title}
                      </li>
                    ))}
                  </ul>
                </div>

                {selectedEstimand.examples && selectedEstimand.examples.python && (
                  <div>
                    <h4 className="font-semibold mb-2">Code Example</h4>
                    <Link to="/terminal" state={{ code: selectedEstimand.examples.python, language: 'Python' }}>
                      <Button className="w-full gap-2">
                        <TerminalIcon className="h-4 w-4" />
                        Run Python in Terminal
                      </Button>
                    </Link>
                  </div>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Index;
