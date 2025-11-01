import Navigation from '@/components/Navigation';
import { useLocation, useNavigate } from 'react-router-dom';
import { estimandsData } from '@/data/estimands';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowLeft, BookOpen, Code, FileText, Presentation, ExternalLink } from 'lucide-react';
import { Separator } from '@/components/ui/separator';

const EstimandOverview = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const params = new URLSearchParams(location.search);
  const estimandId = params.get('id');
  
  const estimand = estimandsData.find(e => e.id === estimandId);
  
  if (!estimand) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="container py-16">
          <p className="text-center text-muted-foreground">Estimand not found</p>
        </div>
      </div>
    );
  }

  const getTierColor = (tier: string) => {
    const colors = {
      Basic: 'bg-tier-basic text-white',
      Intermediate: 'bg-tier-intermediate text-white',
      Advanced: 'bg-tier-advanced text-white',
      Frontier: 'bg-tier-frontier text-white',
    };
    return colors[tier as keyof typeof colors] || 'bg-muted';
  };

  const getFrameworkColor = (framework: string) => {
    const colors = {
      PotentialOutcomes: 'bg-framework-po/10 text-framework-po border-framework-po/20',
      SCM: 'bg-framework-scm/10 text-framework-scm border-framework-scm/20',
      PrincipalStratification: 'bg-framework-ps/10 text-framework-ps border-framework-ps/20',
      ProximalNegativeControl: 'bg-framework-proximal/10 text-framework-proximal border-framework-proximal/20',
      BayesianDecision: 'bg-framework-bayesian/10 text-framework-bayesian border-framework-bayesian/20',
    };
    return colors[framework as keyof typeof colors] || 'bg-muted';
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-12">
        <div className="container">
          <Button 
            variant="ghost" 
            onClick={() => navigate('/estimands')}
            className="mb-4 text-primary-foreground hover:text-primary-foreground hover:bg-primary-foreground/10"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Estimands Library
          </Button>
          
          <div className="flex items-start justify-between gap-4 mb-4">
            <div>
              <h1 className="text-4xl font-bold text-primary-foreground mb-2">
                {estimand.short_name}
              </h1>
              <div className="flex flex-wrap gap-2 mt-4">
                <Badge className={getTierColor(estimand.tier)}>
                  {estimand.tier}
                </Badge>
                <Badge variant="outline" className={`${getFrameworkColor(estimand.framework)} bg-white`}>
                  {estimand.framework}
                </Badge>
                <Badge variant="outline" className="bg-white">
                  {estimand.design.replace(/_/g, ' ')}
                </Badge>
                <Badge variant="outline" className="bg-white">
                  {estimand.estimand_family}
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="py-12">
        <div className="container max-w-5xl">
          <div className="grid gap-6">
            
            {/* Definition */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Definition
                </CardTitle>
              </CardHeader>
              <CardContent>
                <code className="text-lg bg-muted px-3 py-2 rounded block overflow-x-auto">
                  {estimand.definition_tex}
                </code>
              </CardContent>
            </Card>

            {/* Identification Formula */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="h-5 w-5" />
                  Identification Formula
                </CardTitle>
              </CardHeader>
              <CardContent>
                <code className="text-base bg-muted px-3 py-2 rounded block overflow-x-auto">
                  {estimand.identification_formula_tex}
                </code>
                <Separator className="my-4" />
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">Discovery Status:</h4>
                    <Badge variant="secondary">{estimand.discovery_status}</Badge>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">EIF Status:</h4>
                    <Badge variant="secondary">{estimand.eif_status}</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Assumptions */}
            <Card>
              <CardHeader>
                <CardTitle>Key Assumptions</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {estimand.assumptions.map((assumption, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-primary">•</span>
                      <span>{assumption}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Estimators */}
            <Card>
              <CardHeader>
                <CardTitle>Available Estimators</CardTitle>
                <CardDescription>
                  Statistical methods to estimate this causal quantity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {estimand.estimators.map((estimator, idx) => (
                    <Badge key={idx} variant="outline" className="text-sm">
                      {estimator}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Code Examples */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Code className="h-5 w-5" />
                  Code Examples
                </CardTitle>
                <CardDescription>
                  Executable implementations in Python and R
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">Python:</h4>
                    <pre className="bg-muted p-4 rounded text-xs overflow-x-auto">
                      <code>{estimand.examples.python}</code>
                    </pre>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">R:</h4>
                    <pre className="bg-muted p-4 rounded text-xs overflow-x-auto">
                      <code>{estimand.examples.r}</code>
                    </pre>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* References */}
            <Card>
              <CardHeader>
                <CardTitle>Key References</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {estimand.references.map((ref, idx) => (
                    <li key={idx} className="text-sm">
                      <p className="font-medium">{ref.authors} ({ref.year})</p>
                      <p className="text-muted-foreground">{ref.title}</p>
                      <a 
                        href={`https://doi.org/${ref.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary hover:underline text-xs flex items-center gap-1 mt-1"
                      >
                        {ref.doi}
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Action Button */}
            <Card className="border-primary/20 bg-primary/5">
              <CardContent className="pt-6">
                <div className="text-center space-y-4">
                  <h3 className="text-xl font-bold">Ready to Dive Deeper?</h3>
                  <p className="text-muted-foreground">
                    View interactive slides with detailed explanations and visualizations
                  </p>
                  <Button 
                    size="lg"
                    onClick={() => navigate(`/slides?id=${estimand.id}`, { state: { from: '/estimands' } })}
                    className="gap-2"
                  >
                    <Presentation className="h-5 w-5" />
                    View Interactive Slides
                  </Button>
                </div>
              </CardContent>
            </Card>

          </div>
        </div>
      </section>
    </div>
  );
};

export default EstimandOverview;
