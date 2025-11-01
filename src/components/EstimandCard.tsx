import { Estimand } from '@/data/estimands';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Presentation } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface EstimandCardProps {
  estimand: Estimand;
  onClick?: () => void;
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

const getDesignIcon = (design: string) => {
  // RCT designs
  if (design.includes('RCT') || design === 'Cluster_RCT' || design === 'Stepped_Wedge' || design === 'Factorial') {
    return '🎲 RCT';
  }
  // Observational designs
  if (design === 'Cohort' || design === 'Case_Control' || design === 'Cross_Sectional' || design === 'Case_Cohort') {
    return '👁️ Observational';
  }
  // Time-to-event designs
  if (design === 'SCCS' || design === 'Case_Crossover') {
    return '⏱️ TTE';
  }
  // Quasi-experimental
  if (design === 'Regression_Discontinuity' || design === 'Natural_Experiment') {
    return '🔬 Quasi-Exp';
  }
  return '📊 Other';
};

const EstimandCard = ({ estimand, onClick }: EstimandCardProps) => {
  const navigate = useNavigate();
  
  const handleSlidesClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    const index = estimand.id;
    navigate(`/slides?id=${index}`);
  };

  return (
    <Card 
      className="cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5"
      onClick={onClick}
    >
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-lg">{estimand.short_name}</CardTitle>
          <Badge className={getTierColor(estimand.tier)}>{estimand.tier}</Badge>
        </div>
        <CardDescription className="flex flex-wrap gap-1 mt-2">
          <Badge variant="outline" className={getFrameworkColor(estimand.framework)}>
            {estimand.framework}
          </Badge>
          <Badge variant="default" className="bg-blue-600 text-white">
            {getDesignIcon(estimand.design)}
          </Badge>
          <Badge variant="outline">{estimand.design.replace(/_/g, ' ')}</Badge>
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 text-sm">
          <div>
            <span className="font-medium">Family:</span>{' '}
            <span className="text-muted-foreground">{estimand.estimand_family}</span>
          </div>
          <div>
            <span className="font-medium">Definition:</span>{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">{estimand.definition_tex}</code>
          </div>
          <div className="flex gap-2 mt-3">
            <Badge variant="secondary" className="text-xs">
              {estimand.discovery_status}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              EIF: {estimand.eif_status}
            </Badge>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="w-full mt-3 gap-2"
            onClick={handleSlidesClick}
          >
            <Presentation className="h-4 w-4" />
            View Slides
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default EstimandCard;
