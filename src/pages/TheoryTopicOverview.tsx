import Navigation from '@/components/Navigation';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { causalTheory } from '@/data/theory';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { BookOpen, PlayCircle, ArrowLeft, CheckCircle2, Target, BookMarked, Code, FileText } from 'lucide-react';

const allTopics = [...allTheoryTopics, ...causalTheory];

const TheoryTopicOverview = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const topicId = searchParams.get('id');
  
  const topic = allTopics.find(t => t.id === topicId);
  
  if (!topic) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="container py-12 text-center">
          <p className="text-muted-foreground">Topic not found</p>
          <Button onClick={() => navigate('/theory-library')} className="mt-4">
            Back to Library
          </Button>
        </div>
      </div>
    );
  }

  const getTierColor = (tier: string) => {
    const colors: Record<string, string> = {
      'Foundational': 'bg-blue-500/10 text-blue-700 border-blue-500/20',
      'Intermediate': 'bg-green-500/10 text-green-700 border-green-500/20',
      'Advanced': 'bg-orange-500/10 text-orange-700 border-orange-500/20',
      'Frontier': 'bg-purple-500/10 text-purple-700 border-purple-500/20'
    };
    return colors[tier] || 'bg-muted';
  };

  const startLearning = () => {
    navigate(`/theory?id=${topicId}`);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-12">
        <div className="container">
          <Button 
            variant="ghost" 
            onClick={() => navigate('/theory-library')}
            className="mb-4 text-primary-foreground hover:bg-primary-foreground/10"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Theory Library
          </Button>
          
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <Badge className={getTierColor(topic.tier) + ' mb-4'}>
                {topic.tier}
              </Badge>
              <h1 className="text-4xl font-bold text-primary-foreground mb-4">
                {topic.title}
              </h1>
              <p className="text-xl text-primary-foreground/90 max-w-3xl">
                {topic.description}
              </p>
            </div>
            
            <Button 
              size="lg" 
              onClick={startLearning}
              className="gap-2 bg-primary-foreground text-primary hover:bg-primary-foreground/90"
            >
              <PlayCircle className="h-5 w-5" />
              Start Learning
            </Button>
          </div>
        </div>
      </section>

      <div className="container py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content - Table of Contents */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="h-5 w-5" />
                  Table of Contents
                </CardTitle>
                <CardDescription>
                  Structured learning path for this topic
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Learning Objectives */}
                {topic.learningObjectives && topic.learningObjectives.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                      <Target className="h-4 w-4 text-primary" />
                      Learning Objectives
                    </h3>
                    <ul className="space-y-2 ml-6">
                      {topic.learningObjectives.map((obj, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                          <span className="text-muted-foreground">{obj}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <Separator />

                {/* Content Sections */}
                <div>
                  <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                    <BookMarked className="h-4 w-4 text-primary" />
                    Course Sections
                  </h3>
                  <div className="space-y-2 ml-6">
                    {topic.backgroundMotivation && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Background & Motivation</span>
                      </div>
                    )}
                    {topic.historicalContext && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Historical Context</span>
                      </div>
                    )}
                    {topic.conditionsAssumptions && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Conditions & Assumptions</span>
                      </div>
                    )}
                    {topic.keyDefinitions && topic.keyDefinitions.length > 0 && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Key Definitions ({topic.keyDefinitions.length})</span>
                      </div>
                    )}
                    {topic.dataStructureDesign && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Data Structure & Design</span>
                      </div>
                    )}
                    {topic.targetParameter && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Target Parameter</span>
                      </div>
                    )}
                    {topic.identificationStrategy && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Identification Strategy</span>
                      </div>
                    )}
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <div className="w-2 h-2 rounded-full bg-primary" />
                      <span>Core Content</span>
                    </div>
                    {topic.estimationPlan && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Estimation Plan</span>
                      </div>
                    )}
                    {topic.diagnosticsValidation && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Diagnostics & Validation</span>
                      </div>
                    )}
                    {topic.sensitivityRobustness && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Sensitivity & Robustness</span>
                      </div>
                    )}
                    {topic.ethicsGovernance && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span>Ethics & Governance</span>
                      </div>
                    )}
                  </div>
                </div>

                <Separator />

                {/* Code Examples */}
                <div>
                  <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                    <Code className="h-4 w-4 text-primary" />
                    Practical Examples
                  </h3>
                  <div className="space-y-2 ml-6">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <div className="w-2 h-2 rounded-full bg-primary" />
                      <span>Python Implementation</span>
                    </div>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <div className="w-2 h-2 rounded-full bg-primary" />
                      <span>R Implementation</span>
                    </div>
                  </div>
                </div>

                {/* References */}
                {topic.references && topic.references.length > 0 && (
                  <>
                    <Separator />
                    <div>
                      <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                        <FileText className="h-4 w-4 text-primary" />
                        References & Further Reading
                      </h3>
                      <p className="text-sm text-muted-foreground ml-6">
                        {topic.references.length} academic reference{topic.references.length !== 1 ? 's' : ''}
                      </p>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* Start Learning CTA */}
            <Card className="bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-lg mb-2">Ready to dive in?</h3>
                    <p className="text-sm text-muted-foreground">
                      Start the interactive slides to explore this topic in depth
                    </p>
                  </div>
                  <Button size="lg" onClick={startLearning} className="gap-2">
                    <PlayCircle className="h-5 w-5" />
                    Start Learning
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Prerequisites */}
            {topic.prerequisites && topic.prerequisites.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Prerequisites</CardTitle>
                  <CardDescription>
                    Topics to review before starting
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  {topic.prerequisites.map((prereqId, idx) => {
                    const prereqTopic = allTopics.find(t => t.id === prereqId);
                    return prereqTopic ? (
                      <Button
                        key={idx}
                        variant="outline"
                        className="w-full justify-start text-left h-auto py-3"
                        onClick={() => navigate(`/theory-overview?id=${prereqId}`)}
                      >
                        <div className="flex flex-col items-start gap-1">
                          <span className="font-medium">{prereqTopic.title}</span>
                          <Badge variant="secondary" className="text-xs">
                            {prereqTopic.tier}
                          </Badge>
                        </div>
                      </Button>
                    ) : (
                      <div key={idx} className="text-sm text-muted-foreground">
                        {prereqId}
                      </div>
                    );
                  })}
                </CardContent>
              </Card>
            )}

            {/* Quick Info */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Info</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Level:</span>
                  <Badge className={getTierColor(topic.tier)}>
                    {topic.tier}
                  </Badge>
                </div>
                {topic.learningObjectives && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Objectives:</span>
                    <span className="font-medium">{topic.learningObjectives.length}</span>
                  </div>
                )}
                {topic.keyDefinitions && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Key Terms:</span>
                    <span className="font-medium">{topic.keyDefinitions.length}</span>
                  </div>
                )}
                {topic.references && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">References:</span>
                    <span className="font-medium">{topic.references.length}</span>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TheoryTopicOverview;
