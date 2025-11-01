import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { estimandsData } from '@/data/estimands';
import { estimandFamilies } from '@/data/estimandFamilies';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import * as d3 from 'd3';

const NetworkView = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const highlightNodeId = searchParams.get('node'); // Get node to highlight from URL
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [selectedDesign, setSelectedDesign] = useState<string>('all');
  const [selectedFamily, setSelectedFamily] = useState<string>('all');

  // Calculate available filter options based on current selections
  const getFilteredData = () => {
    return estimandsData.filter(e => {
      if (selectedTier !== 'all' && e.tier !== selectedTier) return false;
      if (selectedFramework !== 'all' && e.framework !== selectedFramework) return false;
      if (selectedDesign !== 'all' && e.design !== selectedDesign) return false;
      if (selectedFamily !== 'all' && e.estimand_family !== selectedFamily) return false;
      return true;
    });
  };

  const currentFiltered = getFilteredData();
  
  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(currentFiltered.map(e => e.framework)))];
  const designs = ['all', ...Array.from(new Set(currentFiltered.map(e => e.design))).sort()];
  const families = ['all', ...Array.from(new Set(currentFiltered.map(e => e.estimand_family))).sort()];

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 1800;
    const height = 1200;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('class', 'w-full h-full');

    // Use the filtered estimands
    const filteredEstimands = currentFiltered;

    // Build hierarchy: Root → Framework → Design → Family → Estimands
    const hierarchy: any = { name: 'Causal Inference', type: 'root', children: [] };
    const frameworkMap = new Map();

    filteredEstimands.forEach(est => {
      // Get or create framework
      if (!frameworkMap.has(est.framework)) {
        const fwNode = { name: est.framework, type: 'framework', children: [] };
        frameworkMap.set(est.framework, fwNode);
        hierarchy.children.push(fwNode);
      }
      const fwNode = frameworkMap.get(est.framework);

      // Get or create design
      let designNode = fwNode.children.find((d: any) => d.name === est.design);
      if (!designNode) {
        designNode = { name: est.design, type: 'design', children: [] };
        fwNode.children.push(designNode);
      }

      // Get or create family
      let familyNode = designNode.children.find((f: any) => f.name === est.estimand_family);
      if (!familyNode) {
        familyNode = { name: est.estimand_family, type: 'family', children: [] };
        designNode.children.push(familyNode);
      }

      // Add estimand
      familyNode.children.push({
        name: est.short_name,
        type: 'estimand',
        tier: est.tier,
        id: est.id,
        size: 1
      });
    });

    // Create tree layout
    const root = d3.hierarchy(hierarchy)
      .sum((d: any) => d.size || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const treeLayout = d3.tree<any>()
      .size([height - 100, width - 400])
      .separation((a, b) => (a.parent === b.parent ? 1 : 1.2));

    treeLayout(root);

    // Draw links
    svg.append('g')
      .selectAll('path')
      .data(root.links())
      .join('path')
      .attr('d', d3.linkHorizontal()
        .x((d: any) => d.y + 200)
        .y((d: any) => d.x + 50))
      .attr('fill', 'none')
      .attr('stroke', (d: any) => {
        if (d.target.data.type === 'framework') return 'hsl(215 70% 50%)';
        if (d.target.data.type === 'design') return 'hsl(195 75% 40%)';
        if (d.target.data.type === 'family') return 'hsl(265 60% 45%)';
        return 'hsl(215 15% 60%)';
      })
      .attr('stroke-width', (d: any) => {
        if (d.target.data.type === 'framework') return 3;
        if (d.target.data.type === 'design') return 2.5;
        if (d.target.data.type === 'family') return 2;
        return 1.5;
      })
      .attr('stroke-opacity', 0.6);

    // Draw nodes (include root now)
    const node = svg.append('g')
      .selectAll('g')
      .data(root.descendants())
      .join('g')
      .attr('transform', (d: any) => `translate(${d.y + 200},${d.x + 50})`)
      .style('cursor', 'pointer');

    // Node circles
    node.append('circle')
      .attr('r', (d: any) => {
        if (d.data.type === 'root') return 16;
        if (d.data.type === 'framework') return 12;
        if (d.data.type === 'design') return 10;
        if (d.data.type === 'family') return 8;
        return 6;
      })
      .attr('fill', (d: any) => {
        if (d.data.type === 'root') return 'hsl(280 85% 55%)';
        if (d.data.type === 'framework') return 'hsl(215 70% 50%)';
        if (d.data.type === 'design') return 'hsl(195 75% 40%)';
        if (d.data.type === 'family') return 'hsl(265 60% 45%)';
        // Tier colors for estimands
        switch (d.data.tier) {
          case 'Basic': return 'hsl(142 70% 45%)';
          case 'Intermediate': return 'hsl(215 85% 55%)';
          case 'Advanced': return 'hsl(265 75% 58%)';
          case 'Frontier': return 'hsl(25 90% 55%)';
          default: return 'hsl(215 15% 60%)';
        }
      })
      .attr('stroke', 'white')
      .attr('stroke-width', 2);

    // Node labels
    node.append('text')
      .attr('dy', '0.31em')
      .attr('x', (d: any) => d.children ? -20 : 20)
      .attr('text-anchor', (d: any) => d.children ? 'end' : 'start')
      .text((d: any) => {
        const name = d.data.name;
        if (d.data.type === 'root') return name;
        if (d.data.type === 'framework') return name.replace(/([A-Z])/g, ' $1').trim();
        if (d.data.type === 'design') return name.replace(/_/g, ' ');
        if (d.data.type === 'family') {
          if (name === 'SurvivalTimeToEvent') return 'Survival/Time-to-Event';
          return name.replace(/([A-Z])/g, ' $1').trim();
        }
        return name.length > 40 ? name.substring(0, 38) + '...' : name;
      })
      .attr('font-size', (d: any) => {
        if (d.data.type === 'root') return '16px';
        if (d.data.type === 'framework') return '14px';
        if (d.data.type === 'design') return '12px';
        if (d.data.type === 'family') return '11px';
        return '10px';
      })
      .attr('font-weight', (d: any) => d.data.type === 'estimand' ? 'normal' : 'bold')
      .attr('fill', 'hsl(215 25% 15%)');

    // Click handler - navigate to appropriate content
    node.on('click', (event: any, d: any) => {
      event.stopPropagation();
      if (d.data.type === 'root') {
        navigate('/theory?id=intro-causal-inference', { state: { from: '/network' } });
      } else if (d.data.type === 'estimand') {
        navigate(`/slides?id=${d.data.id}`, { state: { from: '/network' } });
      } else if (d.data.type === 'framework') {
        // Navigate to framework theory slide
        const frameworkIds: Record<string, string> = {
          'PotentialOutcomes': 'framework-potential-outcomes',
          'SCM': 'framework-scm',
          'PrincipalStratification': 'framework-principal-stratification',
          'ProximalNegativeControl': 'framework-proximal-negative-control',
          'BayesianDecision': 'framework-bayesian-decision'
        };
        const fwId = frameworkIds[d.data.name];
        if (fwId) navigate(`/theory?id=${fwId}`, { state: { from: '/network' } });
      } else if (d.data.type === 'design') {
        // Navigate to design theory slide
        const designIds: Record<string, string> = {
          'RCT_Parallel': 'design-rct-parallel',
          'Cohort': 'design-cohort',
          'Cluster_RCT': 'design-cluster-rct',
          'Regression_Discontinuity': 'design-regression-discontinuity',
          'Stepped_Wedge': 'design-stepped-wedge',
          'Encouragement': 'design-encouragement',
          'Case_Control': 'design-case-control',
          'Cross_Sectional': 'design-cross-sectional',
          'Target_Trial_Emulation': 'design-target-trial-emulation',
          'Transport_Frame': 'design-transport-frame',
          'Survey_Multistage': 'design-survey-multistage'
        };
        const designId = designIds[d.data.name];
        if (designId) navigate(`/theory?id=${designId}`, { state: { from: '/network' } });
      } else if (d.data.type === 'family') {
        // Navigate to family theory slide or slides view
        // Create dynamic mapping from family name to family id
        const family = estimandFamilies.find(f => f.name === d.data.name);
        if (family) {
          // If family topic exists, navigate to theory slide
          navigate(`/theory?id=family-${family.id}`, { state: { from: '/network' } });
        } else {
          // Otherwise navigate to slides view with family filter
          navigate(`/slides?family=${d.data.name}`, { state: { from: '/network' } });
        }
      }
    });

    // Hover effects - all nodes now have hover effects
    node.on('mouseenter', function(event: any, d: any) {
      const circle = d3.select(this).select('circle');
      const currentRadius = parseFloat(circle.attr('r'));
      circle.transition().duration(200)
        .attr('r', currentRadius + 2)
        .attr('stroke-width', 3);
    }).on('mouseleave', function(event: any, d: any) {
      const circle = d3.select(this).select('circle');
      let originalRadius = 6;
      if (d.data.type === 'root') originalRadius = 16;
      if (d.data.type === 'framework') originalRadius = 12;
      if (d.data.type === 'design') originalRadius = 10;
      if (d.data.type === 'family') originalRadius = 8;
      
      // Don't shrink if this is the highlighted node
      const isHighlighted = (d.data.type === 'estimand' && d.data.id === highlightNodeId) ||
                           (d.data.type === 'framework' && `framework-${d.data.name.toLowerCase()}` === highlightNodeId) ||
                           (d.data.type === 'design' && `design-${d.data.name.toLowerCase().replace(/_/g, '-')}` === highlightNodeId) ||
                           (d.data.type === 'family' && `family-${d.data.name.toLowerCase().replace(/([A-Z])/g, '-$1').slice(1)}` === highlightNodeId);
      
      if (!isHighlighted) {
        circle.transition().duration(200)
          .attr('r', originalRadius)
          .attr('stroke-width', 2);
      }
    });

    // Highlight the node from URL parameter
    if (highlightNodeId) {
      node.each(function(d: any) {
        const isMatch = (d.data.type === 'estimand' && d.data.id === highlightNodeId) ||
                       (d.data.type === 'root' && highlightNodeId === 'intro-causal-inference') ||
                       (d.data.type === 'framework' && highlightNodeId.startsWith('framework-')) ||
                       (d.data.type === 'design' && highlightNodeId.startsWith('design-')) ||
                       (d.data.type === 'family' && highlightNodeId.startsWith('family-'));
        
        if (isMatch) {
          const circle = d3.select(this).select('circle');
          const text = d3.select(this).select('text');
          
          // Pulse animation
          circle
            .attr('stroke', 'hsl(45 100% 50%)')
            .attr('stroke-width', 4)
            .transition()
            .duration(800)
            .attr('r', parseFloat(circle.attr('r')) + 4)
            .transition()
            .duration(800)
            .attr('r', parseFloat(circle.attr('r')))
            .on('end', function repeat() {
              d3.select(this)
                .transition()
                .duration(800)
                .attr('r', parseFloat(d3.select(this).attr('r')) + 2)
                .transition()
                .duration(800)
                .attr('r', parseFloat(d3.select(this).attr('r')) - 2)
                .on('end', repeat);
            });
          
          text.attr('font-weight', 'bold').attr('fill', 'hsl(45 100% 30%)');
        }
      });
    }

  }, [navigate, selectedTier, selectedFramework, selectedDesign, selectedFamily, highlightNodeId]);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Causal Inference Tree Map</h1>
          <p className="text-muted-foreground max-w-3xl">
            Explore the complete hierarchy of causal inference. Click the root to start with Introduction to Causal Inference, then explore frameworks, study designs, estimand families, and individual estimands. Each node has its own dedicated content.
          </p>
        </div>

        {/* Filters */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Filter by Tier</label>
                <div className="flex flex-wrap gap-2">
                  {tiers.map(tier => (
                    <Badge 
                      key={tier} 
                      variant={selectedTier === tier ? 'default' : 'outline'}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedTier(tier)}
                    >
                      {tier === 'all' ? 'All' : tier}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Filter by Type</label>
                <div className="flex flex-wrap gap-2">
                  {families.map(family => (
                    <Badge 
                      key={family} 
                      variant={selectedFamily === family ? 'default' : 'outline'}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedFamily(family)}
                    >
                      {family === 'all' ? 'All Types' : family === 'SurvivalTimeToEvent' ? 'Survival' : family.replace(/([A-Z])/g, ' $1').trim()}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Filter by Framework</label>
                <div className="flex flex-wrap gap-2">
                  {frameworks.map(fw => (
                    <Badge 
                      key={fw} 
                      variant={selectedFramework === fw ? 'default' : 'outline'}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedFramework(fw)}
                    >
                      {fw === 'all' ? 'All' : fw.replace(/([A-Z])/g, ' $1').trim()}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Filter by Study Design</label>
                <div className="flex flex-wrap gap-2">
                  {designs.map(design => (
                    <Badge 
                      key={design} 
                      variant={selectedDesign === design ? 'default' : 'outline'}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedDesign(design)}
                    >
                      {design === 'all' ? 'All' : design.replace(/_/g, ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Legend */}
        <Card className="mb-6 p-4">
          <h3 className="font-semibold mb-3">Tree Map Legend</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-[hsl(280_85%_55%)]"></div>
              <span className="text-sm">Root (Intro)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-[hsl(215_70%_50%)]"></div>
              <span className="text-sm">Framework</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-[hsl(195_75%_40%)]"></div>
              <span className="text-sm">Study Design</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-[hsl(265_60%_45%)]"></div>
              <span className="text-sm">Estimand Family</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-tier-basic"></div>
              <span className="text-sm">Basic Estimand</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-tier-intermediate"></div>
              <span className="text-sm">Intermediate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-tier-advanced"></div>
              <span className="text-sm">Advanced</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-tier-frontier"></div>
              <span className="text-sm">Frontier</span>
            </div>
          </div>
        </Card>

        <div className="border rounded-lg bg-card p-4 shadow-sm overflow-auto">
          <svg ref={svgRef} className="w-full" style={{ height: '1000px' }}></svg>
        </div>
      </div>
    </div>
  );
};

export default NetworkView;
