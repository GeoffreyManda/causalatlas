import Navigation from '@/components/Navigation';
import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate, Link } from 'react-router-dom';
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BookOpen, ArrowRight, Calculator, Target, Lightbulb, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';

const TheoryLibrary = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  
  // Apply filter from URL param on mount
  useEffect(() => {
    const tier = searchParams.get('tier');
    if (tier) setSelectedTier(tier);
  }, [searchParams]);
  
  const tiers = ['all', 'Foundational', 'Intermediate', 'Advanced', 'Frontier'];
  const categories = ['all', 'math', 'causal'];
  
  // Categorize topics
  const isMathTopic = (topic: any) => {
    const mathKeywords = ['probability', 'distribution', 'expectation', 'variance', 'conditional', 'independence', 'theorem', 'proof', 'algebra', 'calculus', 'statistical', 'regression'];
    const title = topic.title.toLowerCase();
    const desc = topic.description.toLowerCase();
    return mathKeywords.some(keyword => title.includes(keyword) || desc.includes(keyword));
  };
  
  const filteredTopics = allTheoryTopics.filter(topic => {
    if (selectedTier !== 'all' && topic.tier !== selectedTier) return false;
    if (selectedCategory === 'math' && !isMathTopic(topic)) return false;
    if (selectedCategory === 'causal' && isMathTopic(topic)) return false;
    return true;
  });

  const stats = {
    total: allTheoryTopics.length,
    math: allTheoryTopics.filter(t => isMathTopic(t)).length,
    causal: allTheoryTopics.filter(t => !isMathTopic(t)).length,
    tiers: 4
  };

  const getTierColor = (tier: string) => {
    const colors: Record<string, string> = {
      'Foundational': 'bg-blue-500/10 text-blue-700 border-blue-500/20',
      'Intermediate': 'bg-green-500/10 text-green-700 border-green-500/20',
      'Advanced': 'bg-orange-500/10 text-orange-700 border-orange-500/20',
      'Frontier': 'bg-purple-500/10 text-purple-700 border-purple-500/20'
    };
    return colors[tier] || 'bg-muted';
  };

  const handleTopicClick = (topicId: string) => {
    navigate(`/theory-overview?id=${topicId}`);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-20">
        <div className="container">
          <div className="max-w-4xl mx-auto text-center">
            <Badge className="mb-4 text-sm px-4 py-2 bg-white/20 text-white border-white/30">
              Comprehensive Causal Theory
            </Badge>
            <h1 className="text-6xl font-bold text-primary-foreground mb-6 leading-tight">
              Theory Library
            </h1>
            <p className="text-2xl text-primary-foreground/90 mb-8 leading-relaxed">
              Explore mathematical foundations and causal inference theory from basics to advanced topics, 
              with interactive slides and executable examples.
            </p>
            <div className="flex gap-4 justify-center">
              <Link to="/playground">
                <Button size="lg" variant="secondary" className="h-14 px-8 text-lg">
                  <BookOpen className="mr-2 h-5 w-5" />
                  Open Playground
                </Button>
              </Link>
              <Link to="/learning">
                <Button size="lg" variant="outline" className="h-14 px-8 text-lg bg-white/10 hover:bg-white/20 text-white border-white/20">
                  <Target className="mr-2 h-5 w-5" />
                  Learning Hub
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
              <div className="text-4xl font-bold text-primary mb-2">{stats.total}</div>
              <div className="text-sm text-muted-foreground">Total Topics</div>
            </Card>
            <Card className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-4xl font-bold text-blue-600 mb-2">{stats.math}</div>
              <div className="text-sm text-muted-foreground">Math Foundations</div>
            </Card>
            <Card className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-4xl font-bold text-green-600 mb-2">{stats.causal}</div>
              <div className="text-sm text-muted-foreground">Causal Methods</div>
            </Card>
            <Card className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-4xl font-bold text-primary mb-2">{stats.tiers}</div>
              <div className="text-sm text-muted-foreground">Difficulty Tiers</div>
            </Card>
          </div>
        </div>
      </section>

      {/* Category Cards */}
      <section className="py-20">
        <div className="container">
          <h2 className="text-4xl font-bold text-center mb-12">Browse by Category</h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto mb-12">
            
            {/* Math Foundations */}
            <Card 
              className={`p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl ${
                selectedCategory === 'math' 
                  ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white ring-4 ring-blue-300' 
                  : 'bg-gradient-to-br from-card to-blue-500/10'
              } h-full`}
              onClick={() => setSelectedCategory(selectedCategory === 'math' ? 'all' : 'math')}
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className={`h-20 w-20 rounded-2xl flex items-center justify-center shadow-lg ${
                  selectedCategory === 'math' ? 'bg-white/20' : 'bg-blue-500'
                }`}>
                  <Calculator className={`h-10 w-10 ${selectedCategory === 'math' ? 'text-white' : 'text-white'}`} />
                </div>
                <h3 className="text-2xl font-bold">Mathematical Foundations</h3>
                <p className={selectedCategory === 'math' ? 'text-white/90' : 'text-muted-foreground'}>
                  Probability, statistics, regression, and mathematical theory underlying causal inference
                </p>
                <Badge variant={selectedCategory === 'math' ? 'secondary' : 'outline'} className="text-sm">
                  {stats.math} Topics
                </Badge>
              </div>
            </Card>

            {/* Causal Methods */}
            <Card 
              className={`p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl ${
                selectedCategory === 'causal' 
                  ? 'bg-gradient-to-br from-green-500 to-green-600 text-white ring-4 ring-green-300' 
                  : 'bg-gradient-to-br from-card to-green-500/10'
              } h-full`}
              onClick={() => setSelectedCategory(selectedCategory === 'causal' ? 'all' : 'causal')}
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className={`h-20 w-20 rounded-2xl flex items-center justify-center shadow-lg ${
                  selectedCategory === 'causal' ? 'bg-white/20' : 'bg-green-500'
                }`}>
                  <Target className={`h-10 w-10 ${selectedCategory === 'causal' ? 'text-white' : 'text-white'}`} />
                </div>
                <h3 className="text-2xl font-bold">Causal Inference</h3>
                <p className={selectedCategory === 'causal' ? 'text-white/90' : 'text-muted-foreground'}>
                  Frameworks, study designs, identification strategies, and causal estimation methods
                </p>
                <Badge variant={selectedCategory === 'causal' ? 'secondary' : 'outline'} className="text-sm">
                  {stats.causal} Topics
                </Badge>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Filters and Topics */}
      <section className="py-12 bg-muted/30">
        <div className="container">
          {/* Tier Filters */}
          <div className="mb-8 p-6 rounded-lg border bg-card max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold mb-6">Filter by Difficulty</h2>
            <div className="flex flex-wrap gap-2 justify-center">
              {tiers.map((tier) => (
                <Badge
                  key={tier}
                  variant={selectedTier === tier ? "default" : "outline"}
                  className="cursor-pointer px-6 py-3 text-sm hover:scale-105 transition-transform"
                  onClick={() => setSelectedTier(tier)}
                >
                  {tier === 'all' ? 'All Tiers' : tier}
                </Badge>
              ))}
            </div>
            <p className="text-sm text-muted-foreground mt-4 text-center">
              Showing {filteredTopics.length} of {allTheoryTopics.length} topic{filteredTopics.length !== 1 ? 's' : ''}
            </p>
          </div>
          
          {/* Topics Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredTopics.map((topic) => (
              <Card 
                key={topic.id} 
                className="group hover:shadow-lg transition-all duration-300 cursor-pointer hover:scale-[1.02] border-2"
                onClick={() => handleTopicClick(topic.id)}
              >
                <CardHeader>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <Badge className={getTierColor(topic.tier)}>
                      {topic.tier}
                    </Badge>
                    <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                  </div>
                  <CardTitle className="text-xl group-hover:text-primary transition-colors">
                    {topic.title}
                  </CardTitle>
                  <CardDescription className="text-sm">
                    {topic.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {/* Category Badge */}
                    <Badge variant="secondary" className="text-xs">
                      {isMathTopic(topic) ? '📐 Mathematical' : '🎯 Causal'}
                    </Badge>

                    {/* Prerequisites */}
                    {topic.prerequisites && topic.prerequisites.length > 0 && (
                      <div>
                        <p className="text-xs font-medium text-muted-foreground mb-1">Prerequisites:</p>
                        <div className="flex flex-wrap gap-1">
                          {topic.prerequisites.slice(0, 2).map((prereq, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {allTheoryTopics.find(t => t.id === prereq)?.title.slice(0, 20) || prereq}
                            </Badge>
                          ))}
                          {topic.prerequisites.length > 2 && (
                            <Badge variant="outline" className="text-xs">
                              +{topic.prerequisites.length - 2} more
                            </Badge>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Learning Objectives Count */}
                    {topic.learningObjectives && (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <BookOpen className="h-3 w-3" />
                        <span>{topic.learningObjectives.length} learning objectives</span>
                      </div>
                    )}

                    {/* View Overview Button */}
                    <Button 
                      variant="outline" 
                      className="w-full mt-2 group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                      size="sm"
                    >
                      View Learning Path
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          {filteredTopics.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              No topics found for the selected filters
            </div>
          )}
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container">
          <h2 className="text-4xl font-bold text-center mb-12">Learning Features</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            
            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <BookOpen className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Interactive Slides</h3>
                  <p className="text-sm text-muted-foreground">
                    Navigate through topics with beautiful, interactive slide presentations
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Calculator className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Mathematical Rigor</h3>
                  <p className="text-sm text-muted-foreground">
                    Solid mathematical foundations with proofs and derivations
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
                  <h3 className="font-bold mb-2">Clear Prerequisites</h3>
                  <p className="text-sm text-muted-foreground">
                    Know what you need to learn first with prerequisite tracking
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Target className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Learning Objectives</h3>
                  <p className="text-sm text-muted-foreground">
                    Clear goals for what you'll master in each topic
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
                  <h3 className="font-bold mb-2">Progressive Difficulty</h3>
                  <p className="text-sm text-muted-foreground">
                    From foundational to frontier topics, structured paths
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <BookOpen className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Code Examples</h3>
                  <p className="text-sm text-muted-foreground">
                    Python and R implementations for every concept
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

    </div>
  );
};

export default TheoryLibrary;
