import { Estimand } from '@/data/estimands';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import { Play, Copy } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

interface EstimandSlideProps {
  estimand: Estimand;
  isExpertMode: boolean;
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

const BasicModeContent = ({ estimand }: { estimand: Estimand }) => {
  const navigate = useNavigate();
  
  const handleRunCode = (code: string, language: 'python' | 'r') => {
    navigate(`/terminal?code=${encodeURIComponent(code)}&lang=${language}`);
  };

  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    toast.success('Code copied to clipboard');
  };

  const getBasicDescription = () => {
    if (estimand.estimand_family === 'PopulationEffects') {
      return 'This estimand measures the average treatment effect in the population - the difference in outcomes if everyone received treatment versus if no one did.';
    }
    if (estimand.estimand_family === 'DeepRepresentation') {
      return 'This approach uses machine learning to learn representations of the data that are balanced between treatment groups, enabling robust causal effect estimation.';
    }
    if (estimand.estimand_family === 'LongitudinalDynamic') {
      return 'This estimand captures treatment effects over time, accounting for time-varying confounding and treatment history.';
    }
    if (estimand.estimand_family === 'InstrumentalLocal') {
      return 'This estimates the treatment effect for compliers - those who would change their treatment status based on the instrument.';
    }
    if (estimand.estimand_family === 'SurvivalTimeToEvent') {
      return 'This measures the causal effect on survival or time-to-event outcomes, accounting for censoring and competing risks.';
    }
    return 'This estimand represents a causal quantity of interest under specific assumptions and study design.';
  };

  return (
    <div className="space-y-6">
      <div className="prose prose-slate dark:prose-invert max-w-none">
        <p className="text-lg leading-relaxed">{getBasicDescription()}</p>
      </div>

      <div className="bg-muted/50 p-6 rounded-lg">
        <h3 className="font-semibold text-lg mb-3">What You Need</h3>
        <ul className="space-y-2">
          {estimand.assumptions.slice(0, 3).map((assumption, idx) => (
            <li key={idx} className="flex items-start gap-2">
              <span className="text-primary mt-1">•</span>
              <span>{assumption}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="bg-primary/5 p-6 rounded-lg">
        <h3 className="font-semibold text-lg mb-3">How to Estimate</h3>
        <div className="space-y-2">
          {estimand.estimators.slice(0, 3).map((estimator, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <Badge variant="secondary">{estimator}</Badge>
            </div>
          ))}
        </div>
      </div>

      {estimand.examples && estimand.examples.python && (
        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-lg">Try It Out (Python)</h3>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleCopyCode(estimand.examples.python)}
              >
                <Copy className="h-4 w-4" />
              </Button>
              <Button
                size="sm"
                onClick={() => handleRunCode(estimand.examples.python, 'python')}
              >
                <Play className="h-4 w-4 mr-2" />
                Run in Terminal
              </Button>
            </div>
          </div>
          <pre className="text-xs bg-muted p-4 rounded overflow-x-auto">
            <code>{estimand.examples.python.split('\n').slice(0, 10).join('\n')}...</code>
          </pre>
        </div>
      )}
    </div>
  );
};

const ExpertModeContent = ({ estimand }: { estimand: Estimand }) => {
  const navigate = useNavigate();

  const handleRunCode = (code: string, language: 'python' | 'r') => {
    navigate(`/terminal?code=${encodeURIComponent(code)}&lang=${language}`);
  };

  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    toast.success('Code copied to clipboard');
  };

  return (
    <div className="space-y-6">
      {/* Formal Definition */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Formal Definition</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-muted/50 p-4 rounded-lg overflow-x-auto">
            <BlockMath math={estimand.definition_tex} />
          </div>
        </CardContent>
      </Card>

      {/* Assumptions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Identification Assumptions</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {estimand.assumptions.map((assumption, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-primary font-bold mt-1">{idx + 1}.</span>
                <span className="font-mono text-sm">{assumption}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Identification Formula */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Identification Formula</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-muted/50 p-4 rounded-lg overflow-x-auto">
            <BlockMath math={estimand.identification_formula_tex} />
          </div>
        </CardContent>
      </Card>

      {/* Estimators */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Available Estimators</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {estimand.estimators.map((estimator, idx) => (
              <Badge key={idx} variant="outline" className="font-mono">
                {estimator}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Technical Details */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Technical Properties</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-muted-foreground">Discovery Status</span>
              <div className="font-semibold mt-1 capitalize">{estimand.discovery_status.replace('_', ' ')}</div>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">EIF Status</span>
              <div className="font-semibold mt-1 capitalize">{estimand.eif_status.replace('_', ' ')}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* References */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Key References</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {estimand.references.map((ref, idx) => (
              <li key={idx} className="text-sm">
                <div className="font-medium">{ref.authors} ({ref.year})</div>
                <div className="text-muted-foreground">{ref.title}</div>
                {ref.doi && (
                  <div className="text-xs text-primary mt-1">DOI: {ref.doi}</div>
                )}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Full Code Examples */}
      {estimand.examples && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Implementation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {estimand.examples.python && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Badge>Python</Badge>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleCopyCode(estimand.examples.python)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => handleRunCode(estimand.examples.python, 'python')}
                    >
                      <Play className="h-4 w-4 mr-2" />
                      Run
                    </Button>
                  </div>
                </div>
                <pre className="text-xs bg-muted p-4 rounded overflow-x-auto">
                  <code>{estimand.examples.python}</code>
                </pre>
              </div>
            )}
            {estimand.examples.r && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Badge>R</Badge>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleCopyCode(estimand.examples.r)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => handleRunCode(estimand.examples.r, 'r')}
                    >
                      <Play className="h-4 w-4 mr-2" />
                      Run
                    </Button>
                  </div>
                </div>
                <pre className="text-xs bg-muted p-4 rounded overflow-x-auto">
                  <code>{estimand.examples.r}</code>
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const EstimandSlide = ({ estimand, isExpertMode }: EstimandSlideProps) => {
  return (
    <Card className="min-h-[600px]">
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-2">
            <CardTitle className="text-3xl">{estimand.short_name}</CardTitle>
            <div className="flex flex-wrap gap-2">
              <Badge className={getTierColor(estimand.tier)}>{estimand.tier}</Badge>
              <Badge variant="outline" className={getFrameworkColor(estimand.framework)}>
                {estimand.framework}
              </Badge>
              <Badge variant="outline">{estimand.design}</Badge>
              <Badge variant="secondary">{estimand.estimand_family}</Badge>
            </div>
          </div>
        </div>
      </CardHeader>
      <Separator />
      <CardContent className="pt-6">
        {isExpertMode ? (
          <ExpertModeContent estimand={estimand} />
        ) : (
          <BasicModeContent estimand={estimand} />
        )}
      </CardContent>
    </Card>
  );
};

export default EstimandSlide;
