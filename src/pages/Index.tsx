import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Link } from 'react-router-dom';
import { GraduationCap, Target, Network, BookOpen, Code, Lightbulb, TrendingUp, Database } from 'lucide-react';
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { estimandsData } from '@/data/estimands';
import { estimandFamilies } from '@/data/estimandFamilies';

const Index = () => {
  const stats = {
    theory: allTheoryTopics.length,
    estimands: estimandsData.length,
    families: estimandFamilies.length,
    tiers: 4
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-20">
        <div className="container">
          <div className="max-w-4xl mx-auto text-center">
            <Badge className="mb-4 text-sm px-4 py-2 bg-white/20 text-white border-white/30">
              Publication-Grade Causal Inference
            </Badge>
            <h1 className="text-6xl font-bold text-primary-foreground mb-6 leading-tight">
              Causal Estimand Atlas
            </h1>
            <p className="text-2xl text-primary-foreground/90 mb-8 leading-relaxed">
              Navigate the complete landscape of causal inference from foundational theory to frontier methods, 
              with interactive visualizations and executable code.
            </p>
            <div className="flex gap-4 justify-center">
              <Link to="/learning">
                <Button size="lg" variant="secondary" className="h-14 px-8 text-lg">
                  <GraduationCap className="mr-2 h-5 w-5" />
                  Start Learning
                </Button>
              </Link>
              <Link to="/network">
                <Button size="lg" variant="outline" className="h-14 px-8 text-lg bg-white/10 hover:bg-white/20 text-white border-white/20">
                  <Network className="mr-2 h-5 w-5" />
                  Explore Network
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 bg-muted/30">
        <div className="container">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
            <Card className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-4xl font-bold text-primary mb-2">{stats.theory}+</div>
              <div className="text-sm text-muted-foreground">Theory Topics</div>
            </Card>
            <Card className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-4xl font-bold text-primary mb-2">{stats.estimands}+</div>
              <div className="text-sm text-muted-foreground">Estimands</div>
            </Card>
            <Card className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-4xl font-bold text-primary mb-2">{stats.families}+</div>
              <div className="text-sm text-muted-foreground">Estimand Families</div>
            </Card>
            <Card className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-4xl font-bold text-primary mb-2">{stats.tiers}</div>
              <div className="text-sm text-muted-foreground">Complexity Tiers</div>
            </Card>
          </div>
        </div>
      </section>

      {/* Main Navigation Cards */}
      <section className="py-20">
        <div className="container">
          <h2 className="text-4xl font-bold text-center mb-12">Explore the Atlas</h2>
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            
            {/* Theory Hub */}
            <Link to="/theory">
              <Card className="p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card to-tier-basic/10 h-full">
                <div className="flex flex-col items-center text-center space-y-4">
                  <div className="h-20 w-20 rounded-2xl bg-tier-basic flex items-center justify-center shadow-lg">
                    <GraduationCap className="h-10 w-10 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold">Theory Hub</h3>
                  <p className="text-muted-foreground">
                    Master foundational concepts: DAGs, potential outcomes, identification strategies
                  </p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    <Badge variant="outline" className="text-xs">Foundational</Badge>
                    <Badge variant="outline" className="text-xs">Intermediate</Badge>
                    <Badge variant="outline" className="text-xs">Advanced</Badge>
                  </div>
                </div>
              </Card>
            </Link>

            {/* Estimands Library */}
            <Link to="/estimands">
              <Card className="p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card to-tier-intermediate/10 h-full">
                <div className="flex flex-col items-center text-center space-y-4">
                  <div className="h-20 w-20 rounded-2xl bg-tier-intermediate flex items-center justify-center shadow-lg">
                    <Target className="h-10 w-10 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold">Estimands Library</h3>
                  <p className="text-muted-foreground">
                    Explore specific estimands from ATE to frontier methods with executable code
                  </p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    <Badge variant="outline" className="text-xs">Population</Badge>
                    <Badge variant="outline" className="text-xs">Mediation</Badge>
                    <Badge variant="outline" className="text-xs">Survival</Badge>
                  </div>
                </div>
              </Card>
            </Link>

            {/* Network View */}
            <Link to="/network">
              <Card className="p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card to-tier-advanced/10 h-full">
                <div className="flex flex-col items-center text-center space-y-4">
                  <div className="h-20 w-20 rounded-2xl bg-tier-advanced flex items-center justify-center shadow-lg">
                    <Network className="h-10 w-10 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold">Network View</h3>
                  <p className="text-muted-foreground">
                    Visualize connections between frameworks, designs, and estimands
                  </p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    <Badge variant="outline" className="text-xs">Interactive</Badge>
                    <Badge variant="outline" className="text-xs">Visual</Badge>
                    <Badge variant="outline" className="text-xs">Connected</Badge>
                  </div>
                </div>
              </Card>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-muted/30">
        <div className="container">
          <h2 className="text-4xl font-bold text-center mb-12">Why This Atlas?</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            
            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <BookOpen className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Comprehensive Coverage</h3>
                  <p className="text-sm text-muted-foreground">
                    All major frameworks, designs, and estimands in one place
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Code className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Executable Code</h3>
                  <p className="text-sm text-muted-foreground">
                    Python & R examples for every estimand, ready to run
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Lightbulb className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Interactive Learning</h3>
                  <p className="text-sm text-muted-foreground">
                    Visual slides, network graphs, and guided tutorials
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <TrendingUp className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Progressive Depth</h3>
                  <p className="text-sm text-muted-foreground">
                    From foundational to frontier, structured learning paths
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Quick Access Section */}
      <section className="py-20">
        <div className="container">
          <h2 className="text-4xl font-bold text-center mb-4">Quick Access</h2>
          <p className="text-center text-muted-foreground mb-12 text-lg">
            Jump directly to what you need
          </p>
          <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            
            <Link to="/slides">
              <Card className="p-6 cursor-pointer hover:shadow-xl transition-all hover:scale-[1.02]">
                <div className="flex items-center gap-4">
                  <div className="h-14 w-14 rounded-xl bg-primary/10 flex items-center justify-center">
                    <BookOpen className="h-7 w-7 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-1">Generated Slides</h3>
                    <p className="text-sm text-muted-foreground">View presentation-ready slide decks</p>
                  </div>
                </div>
              </Card>
            </Link>

            <Link to="/terminal">
              <Card className="p-6 cursor-pointer hover:shadow-xl transition-all hover:scale-[1.02]">
                <div className="flex items-center gap-4">
                  <div className="h-14 w-14 rounded-xl bg-primary/10 flex items-center justify-center">
                    <Database className="h-7 w-7 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-1">Terminal View</h3>
                    <p className="text-sm text-muted-foreground">Interactive terminal-style exploration</p>
                  </div>
                </div>
              </Card>
            </Link>
          </div>
        </div>
      </section>

    </div>
  );
};

export default Index;
