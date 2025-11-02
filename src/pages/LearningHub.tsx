import Navigation from '@/components/Navigation';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { GraduationCap, Target, BookOpen, Code, Calculator, Lightbulb, TrendingUp, Network } from 'lucide-react';
import { useNavigate, Link } from 'react-router-dom';
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { estimandsData } from '@/data/estimands';

const LearningHub = () => {
  const navigate = useNavigate();
  
  // Categorize topics using same strict logic
  const isMathTopic = (topic: any) => {
    const title = topic.title.toLowerCase();
    const desc = topic.description.toLowerCase();
    const content = title + ' ' + desc;
    
    // Causal inference keywords - if any of these appear, it's NOT a math topic
    const causalKeywords = [
      'causal', 'dag', 'd-separation', 'do-calculus', 'counterfactual', 'potential outcome',
      'instrumental variable', 'rdd', 'regression discontinuity', 'difference-in-differences',
      'did', 'matching', 'propensity score', 'ate', 'att', 'atet', 'cate', 'late',
      'backdoor', 'front-door', 'collider', 'confounder', 'mediation', 'graphoid',
      'intervention', 'treatment', 'estimand', 'identification', 'unconfoundedness',
      'ignorability', 'exogeneity', 'endogeneity', 'selection bias', 'omitted variable',
      'synthetic control', 'event study', 'parallel trends', 'common support',
      'balancing score', 'overlap', 'positivity', 'sutva', 'consistency', 'framework',
      'study design', 'observational', 'experimental'
    ];
    
    if (causalKeywords.some(keyword => content.includes(keyword))) {
      return false;
    }
    
    const pureMathKeywords = [
      'measure theory', 'probability space', 'sigma-algebra', 'measurable',
      'expectation', 'moment', 'variance', 'covariance', 'correlation',
      'distribution', 'density', 'cumulative distribution',
      'convergence', 'limit theorem', 'central limit', 'law of large numbers',
      'moment generating', 'characteristic function', 'martingale',
      'maximum likelihood', 'bayesian', 'hypothesis test', 'fisher information',
      'confidence interval', 'p-value', 'power analysis', 'likelihood ratio',
      'linear algebra', 'matrix', 'eigenvalue', 'optimization',
      'bootstrap', 'cross-validation', 'regularization', 'lasso', 'ridge',
      'statistical inference', 'estimator properties', 'bias', 'consistency',
      'asymptotic', 'efficient', 'unbiased', 'sufficient statistic',
      'regression analysis', 'linear model', 'generalized linear',
      'semiparametric', 'nonparametric', 'kernel', 'spline',
      'random variable', 'stochastic', 'markov', 'process'
    ];
    
    return pureMathKeywords.some(keyword => content.includes(keyword));
  };

  const stats = {
    total: allTheoryTopics.length,
    math: allTheoryTopics.filter(t => isMathTopic(t)).length,
    causal: allTheoryTopics.filter(t => !isMathTopic(t)).length,
    estimands: estimandsData.length
  };
  
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-20">
        <div className="container">
          <div className="max-w-4xl mx-auto text-center">
            <Badge className="mb-4 text-sm px-4 py-2 bg-white/20 text-white border-white/30">
              Structured Learning Paths
            </Badge>
            <h1 className="text-6xl font-bold text-primary-foreground mb-6 leading-tight">
              Learning Hub
            </h1>
            <p className="text-2xl text-primary-foreground/90 mb-8 leading-relaxed">
              Master causal inference through structured paths: Start with mathematical foundations, 
              build causal intuition, then apply specific estimands to real problems.
            </p>
            <div className="flex gap-4 justify-center">
              <Link to="/playground">
                <Button size="lg" variant="secondary" className="h-14 px-8 text-lg">
                  <Code className="mr-2 h-5 w-5" />
                  Try Playground
                </Button>
              </Link>
              <Link to="/network">
                <Button size="lg" variant="outline" className="h-14 px-8 text-lg bg-white/10 hover:bg-white/20 text-white border-white/20">
                  <Network className="mr-2 h-5 w-5" />
                  Network View
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
              <div className="text-sm text-muted-foreground">Theory Topics</div>
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
              <div className="text-4xl font-bold text-primary mb-2">{stats.estimands}</div>
              <div className="text-sm text-muted-foreground">Estimands</div>
            </Card>
          </div>
        </div>
      </section>

      {/* Learning Paths */}
      <section className="py-20">
        <div className="container">
          <h2 className="text-4xl font-bold text-center mb-12">Choose Your Learning Path</h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto mb-12">
            
            {/* Math Foundations Path */}
            <Card 
              onClick={() => navigate('/theory-library?category=math')}
              className="p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card to-blue-500/10 h-full"
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className="h-20 w-20 rounded-2xl bg-blue-500 flex items-center justify-center shadow-lg">
                  <Calculator className="h-10 w-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold">Mathematical Foundations</h3>
                <p className="text-muted-foreground">
                  Build your statistical and mathematical foundation with probability theory, distributions, and inference methods
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  <Badge variant="outline" className="text-xs">Probability Theory</Badge>
                  <Badge variant="outline" className="text-xs">Statistical Inference</Badge>
                  <Badge variant="outline" className="text-xs">Regression</Badge>
                </div>
                <Badge className="text-sm bg-blue-500">
                  {stats.math} Topics
                </Badge>
              </div>
            </Card>

            {/* Causal Inference Path */}
            <Card 
              onClick={() => navigate('/theory-library?category=causal')}
              className="p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card to-green-500/10 h-full"
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className="h-20 w-20 rounded-2xl bg-green-500 flex items-center justify-center shadow-lg">
                  <Target className="h-10 w-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold">Causal Inference Theory</h3>
                <p className="text-muted-foreground">
                  Master causal frameworks, DAGs, identification strategies, and study designs for causal questions
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  <Badge variant="outline" className="text-xs">Causal DAGs</Badge>
                  <Badge variant="outline" className="text-xs">Identification</Badge>
                  <Badge variant="outline" className="text-xs">Frameworks</Badge>
                </div>
                <Badge className="text-sm bg-green-500">
                  {stats.causal} Topics
                </Badge>
              </div>
            </Card>

            {/* Estimands Path */}
            <Card 
              onClick={() => navigate('/estimands')}
              className="p-8 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card to-tier-intermediate/10 h-full md:col-span-2"
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className="h-20 w-20 rounded-2xl bg-tier-intermediate flex items-center justify-center shadow-lg">
                  <Code className="h-10 w-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold">Applied Estimands & Methods</h3>
                <p className="text-muted-foreground max-w-2xl">
                  Apply your knowledge with specific estimands from ATE to frontier methods, complete with Python and R code examples
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  <Badge variant="outline" className="text-xs">Population Effects</Badge>
                  <Badge variant="outline" className="text-xs">Mediation</Badge>
                  <Badge variant="outline" className="text-xs">Instrumental Variables</Badge>
                  <Badge variant="outline" className="text-xs">Machine Learning</Badge>
                </div>
                <Badge className="text-sm bg-tier-intermediate">
                  {stats.estimands} Estimands
                </Badge>
              </div>
            </Card>

          </div>
        </div>
      </section>

      {/* Explanatory Section */}
      <section className="py-20 bg-muted/30">
        <div className="container">
          <div className="max-w-4xl mx-auto">
            <Card className="p-8">
              <h3 className="text-3xl font-bold mb-6 text-center">Understanding the Hierarchy</h3>
              <div className="space-y-4 text-muted-foreground">
                <p className="text-lg">
                  <strong className="text-foreground">Mathematical Foundations</strong> provide the statistical and probabilistic tools needed for causal inference: probability theory, distributions, regression, and inference methods.
                </p>
                <p className="text-lg">
                  <strong className="text-foreground">Causal Theory</strong> introduces frameworks for thinking about causation: Potential Outcomes, DAGs, do-calculus, and identification strategies that tell us when we can estimate causal effects.
                </p>
                <p className="text-lg">
                  <strong className="text-foreground">Estimands</strong> are the specific causal quantities we want to estimate (e.g., ATE, CATE). They define <em>what</em> we're trying to measure.
                </p>
                <p className="text-lg">
                  <strong className="text-foreground">Estimators</strong> are the statistical methods or algorithms used to estimate the estimands from observed data (e.g., IPW, TMLE, G-computation).
                </p>
                <p className="text-lg font-semibold text-foreground mt-6">
                  This learning hub guides you from foundations → theory → estimands → code, ensuring you understand both the "what" and the "how" of causal inference.
                </p>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container">
          <h2 className="text-4xl font-bold text-center mb-12">Learning Features</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            
            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <BookOpen className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-bold mb-2">Interactive Slides</h3>
                  <p className="text-sm text-muted-foreground">
                    Learn with beautiful presentation slides for every topic
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
                    Python & R examples you can run in the playground
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
                    Know exactly what to learn first with prerequisite tracking
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
                    From foundational to frontier, structured learning
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

export default LearningHub;
