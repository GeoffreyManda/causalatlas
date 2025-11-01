import { useNavigate } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { Card } from '@/components/ui/card';
import { GraduationCap, Target, BookOpen, Code } from 'lucide-react';

const LearningHub = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-16">
        <div className="container text-center">
          <h1 className="text-5xl font-bold text-primary-foreground mb-4">
            Structured Learning Hub
          </h1>
          <p className="text-xl text-primary-foreground/90 max-w-3xl mx-auto">
            Master causal inference through a structured approach: Start with foundational theory, then explore specific estimands and their applications.
          </p>
        </div>
      </section>

      {/* Learning Paths */}
      <section className="py-20">
        <div className="container">
          <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
            
            {/* Causal Theory Path */}
            <Card 
              onClick={() => navigate('/theory')}
              className="p-12 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card via-card to-tier-basic/10"
            >
              <div className="flex flex-col items-center text-center space-y-6">
                <div className="h-24 w-24 rounded-2xl bg-tier-basic flex items-center justify-center shadow-lg">
                  <GraduationCap className="h-12 w-12 text-white" />
                </div>
                <div>
                  <h2 className="text-3xl font-bold mb-4">Causal Theory</h2>
                  <p className="text-muted-foreground text-lg mb-6">
                    Learn the foundational frameworks: Potential Outcomes, Structural Causal Models, and more
                  </p>
                  <div className="flex flex-col gap-2 text-sm">
                    <div className="flex items-center gap-2 justify-center">
                      <BookOpen className="h-4 w-4 text-tier-basic" />
                      <span>Introduction to Causal Inference</span>
                    </div>
                    <div className="flex items-center gap-2 justify-center">
                      <BookOpen className="h-4 w-4 text-tier-basic" />
                      <span>Framework-Specific Deep Dives</span>
                    </div>
                    <div className="flex items-center gap-2 justify-center">
                      <BookOpen className="h-4 w-4 text-tier-basic" />
                      <span>Mathematical Foundations</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            {/* Causal Estimands Path */}
            <Card 
              onClick={() => navigate('/slides')}
              className="p-12 cursor-pointer hover:scale-105 transition-all hover:shadow-2xl bg-gradient-to-br from-card via-card to-tier-intermediate/10"
            >
              <div className="flex flex-col items-center text-center space-y-6">
                <div className="h-24 w-24 rounded-2xl bg-tier-intermediate flex items-center justify-center shadow-lg">
                  <Target className="h-12 w-12 text-white" />
                </div>
                <div>
                  <h2 className="text-3xl font-bold mb-4">Causal Estimands</h2>
                  <p className="text-muted-foreground text-lg mb-6">
                    Explore specific estimands from ATE to frontier methods, with executable code examples
                  </p>
                  <div className="flex flex-col gap-2 text-sm">
                    <div className="flex items-center gap-2 justify-center">
                      <Code className="h-4 w-4 text-tier-intermediate" />
                      <span>Population Effects (ATE, ATT, CATE)</span>
                    </div>
                    <div className="flex items-center gap-2 justify-center">
                      <Code className="h-4 w-4 text-tier-intermediate" />
                      <span>Mediation & Instrumental Variables</span>
                    </div>
                    <div className="flex items-center gap-2 justify-center">
                      <Code className="h-4 w-4 text-tier-intermediate" />
                      <span>Frontier Methods & Deep Learning</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>

          </div>

          {/* Explanatory Section */}
          <div className="mt-20 max-w-4xl mx-auto">
            <Card className="p-8 bg-muted/30">
              <h3 className="text-2xl font-bold mb-4 text-center">Understanding the Hierarchy</h3>
              <div className="space-y-4 text-muted-foreground">
                <p className="text-lg">
                  <strong className="text-foreground">Estimands</strong> are the theoretical causal quantities we want to estimate (e.g., the average treatment effect). They define <em>what</em> we're trying to measure.
                </p>
                <p className="text-lg">
                  <strong className="text-foreground">Estimators</strong> are the statistical methods or algorithms used to estimate the estimands from observed data (e.g., inverse probability weighting, TMLE).
                </p>
                <p className="text-lg">
                  <strong className="text-foreground">Estimates</strong> are the actual numerical values we compute by applying estimators to real data.
                </p>
                <p className="text-lg font-semibold text-foreground mt-6">
                  This atlas helps you navigate from theory → estimands → estimators → executable code, ensuring you understand both the "what" and the "how" of causal inference.
                </p>
              </div>
            </Card>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LearningHub;
