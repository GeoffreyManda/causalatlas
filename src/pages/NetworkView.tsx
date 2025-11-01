import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { estimandsData } from '@/data/estimands';
import { Button } from '@/components/ui/button';
import * as d3 from 'd3';

const NetworkView = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const navigate = useNavigate();
  const [layoutMode, setLayoutMode] = useState<'static' | 'dynamic'>('static');

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 1400;
    const height = 900;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('class', 'w-full h-full');

    // Create HIERARCHICAL nodes
    const nodes: any[] = [];
    const links: any[] = [];

    // Layer 1: Frameworks (top)
    const frameworks = Array.from(new Set(estimandsData.map(e => e.framework)));
    frameworks.forEach((fw, i) => {
      nodes.push({ 
        id: `fw_${fw}`, 
        label: fw, 
        type: 'framework', 
        x: 200 + i * 250,
        y: 100,
        fx: 200 + i * 250, // Fixed position
        fy: 100
      });
    });

    // Layer 2: Estimands grouped by tier (hierarchical tiers)
    const tierY = { Basic: 300, Intermediate: 500, Advanced: 700, Frontier: 850 };
    const tierCounts = { Basic: 0, Intermediate: 0, Advanced: 0, Frontier: 0 };

    estimandsData.forEach((estimand) => {
      const fwIndex = frameworks.indexOf(estimand.framework);
      const tierCount = tierCounts[estimand.tier as keyof typeof tierCounts];
      
      nodes.push({
        id: estimand.id,
        label: estimand.short_name,
        type: 'estimand',
        tier: estimand.tier,
        framework: estimand.framework,
        design: estimand.design,
        x: 200 + fwIndex * 250 + (tierCount % 3 - 1) * 80,
        y: tierY[estimand.tier as keyof typeof tierY]
      });
      
      tierCounts[estimand.tier as keyof typeof tierCounts]++;
      
      // Hierarchical link to framework
      links.push({
        source: `fw_${estimand.framework}`,
        target: estimand.id,
        type: 'belongs_to'
      });
    });

    // Add causal dependencies (theory -> applications)
    const dependencies = [
      { from: 'dags_scm', to: 'ate' },
      { from: 'dags_scm', to: 'att' },
      { from: 'd_separation', to: 'cate' },
      { from: 'do_calculus', to: 'ate' },
      { from: 'do_calculus', to: 'nde' },
      { from: 'intro_causality', to: 'ate' },
      { from: 'ate', to: 'att' },
      { from: 'ate', to: 'cate' },
      { from: 'msm', to: 'doubleml' },
      { from: 'late', to: 'nde' },
    ];

    dependencies.forEach(dep => {
      const sourceNode = nodes.find(n => n.id === dep.from);
      const targetNode = nodes.find(n => n.id === dep.to);
      if (sourceNode && targetNode) {
        links.push({
          source: dep.from,
          target: dep.to,
          type: 'causal_dependency'
        });
      }
    });

    // Force simulation with conditional positioning
    const simulation = d3.forceSimulation(nodes)
      .force('x', d3.forceX((d: any) => d.x).strength(layoutMode === 'static' ? 0.9 : 0.3))
      .force('y', d3.forceY((d: any) => d.y).strength(layoutMode === 'static' ? 0.9 : 0.3))
      .force('collision', d3.forceCollide().radius(50))
      .force('charge', d3.forceManyBody().strength(layoutMode === 'dynamic' ? -400 : -100))
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance((d: any) => d.type === 'causal_dependency' ? 100 : 150)
        .strength(layoutMode === 'dynamic' ? 0.5 : 0.3));


    // Add links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', 'hsl(215 15% 88%)')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6);

    // Add nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add circles
    node.append('circle')
      .attr('r', (d: any) => d.type === 'framework' ? 30 : 20)
      .attr('fill', (d: any) => {
        if (d.type === 'framework') return 'hsl(215 60% 25%)';
        switch (d.tier) {
          case 'Basic': return 'hsl(142 70% 45%)';
          case 'Intermediate': return 'hsl(215 85% 55%)';
          case 'Advanced': return 'hsl(265 75% 58%)';
          case 'Frontier': return 'hsl(25 90% 55%)';
          default: return 'hsl(215 15% 60%)';
        }
      })
      .attr('stroke', 'white')
      .attr('stroke-width', 2);

    // Add labels
    node.append('text')
      .text((d: any) => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', (d: any) => d.type === 'framework' ? 50 : 35)
      .attr('font-size', (d: any) => d.type === 'framework' ? '14px' : '11px')
      .attr('font-weight', (d: any) => d.type === 'framework' ? 'bold' : 'normal')
      .attr('fill', 'hsl(215 25% 15%)');

    // Click to open slide
    node.on('click', (event: any, d: any) => {
      event.stopPropagation();
      if (d.type === 'estimand') {
        navigate(`/slides?id=${d.id}`);
      }
    });

    // Update positions
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return () => {
      simulation.stop();
    };
  }, [layoutMode]);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Causal Inference Organogram</h1>
            <p className="text-muted-foreground">
              Interactive network showing frameworks, estimands, and difficulty levels • Click nodes to explore slides
            </p>
          </div>
          <div className="flex gap-2">
            <Button 
              variant={layoutMode === 'static' ? 'default' : 'outline'}
              onClick={() => setLayoutMode('static')}
            >
              Static Layout
            </Button>
            <Button 
              variant={layoutMode === 'dynamic' ? 'default' : 'outline'}
              onClick={() => setLayoutMode('dynamic')}
            >
              Dynamic Layout
            </Button>
          </div>
        </div>

        {/* Legend */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <h3 className="font-semibold mb-3">Legend</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded-full bg-tier-basic"></div>
              <span className="text-sm">Basic</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded-full bg-tier-intermediate"></div>
              <span className="text-sm">Intermediate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded-full bg-tier-advanced"></div>
              <span className="text-sm">Advanced</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded-full bg-tier-frontier"></div>
              <span className="text-sm">Frontier</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded-full bg-primary"></div>
              <span className="text-sm">Framework</span>
            </div>
          </div>
        </div>

        {/* Network Visualization */}
        <div className="border rounded-lg bg-white p-4">
          <svg ref={svgRef} className="w-full" style={{ height: '800px' }}></svg>
        </div>

        <p className="text-sm text-muted-foreground mt-4 text-center">
          {layoutMode === 'static' 
            ? 'Hierarchical layout: Frameworks → Estimands by Tier • Drag nodes to adjust • Click to view slides'
            : 'Force-directed layout: Dynamic clustering by relationships • Drag nodes to explore • Click to view slides'
          }
        </p>
      </div>
    </div>
  );
};

export default NetworkView;
