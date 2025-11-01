import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { estimandsData } from '@/data/estimands';
import { causalTheory } from '@/data/theory';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import * as d3 from 'd3';

const NetworkView = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const navigate = useNavigate();
  const [layoutMode, setLayoutMode] = useState<'static' | 'dynamic'>('static');

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 1600;
    const height = 1000;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('class', 'w-full h-full');

    // Create HIERARCHICAL STRUCTURE
    const nodes: any[] = [];
    const links: any[] = [];

    // Layer 1: Theory Topics (top tier)
    const foundationalTheory = causalTheory.filter(t => t.tier === 'Foundational').slice(0, 4);
    foundationalTheory.forEach((theory, i) => {
      nodes.push({
        id: `theory_${theory.id}`,
        label: theory.title,
        type: 'theory',
        tier: theory.tier,
        originalId: theory.id,
        x: 200 + i * 350,
        y: 80,
        fx: layoutMode === 'static' ? 200 + i * 350 : null,
        fy: layoutMode === 'static' ? 80 : null
      });
    });

    // Layer 2: Frameworks (middle tier)
    const frameworks = Array.from(new Set(estimandsData.map(e => e.framework)));
    frameworks.forEach((fw, i) => {
      nodes.push({
        id: `fw_${fw}`,
        label: fw.replace(/([A-Z])/g, ' $1').trim(),
        type: 'framework',
        x: 150 + i * 300,
        y: 280,
        fx: layoutMode === 'static' ? 150 + i * 300 : null,
        fy: layoutMode === 'static' ? 280 : null
      });
    });

    // Layer 3: Estimands grouped by tier (bottom tiers)
    const tierY = { Basic: 500, Intermediate: 650, Advanced: 800, Frontier: 920 };
    const tierXOffset: any = {};
    
    frameworks.forEach(fw => {
      tierXOffset[fw] = { Basic: 0, Intermediate: 0, Advanced: 0, Frontier: 0 };
    });

    estimandsData.forEach((estimand) => {
      const fwIndex = frameworks.indexOf(estimand.framework);
      const offset = tierXOffset[estimand.framework][estimand.tier];
      
      nodes.push({
        id: estimand.id,
        label: estimand.short_name.length > 30 ? estimand.short_name.substring(0, 28) + '...' : estimand.short_name,
        type: 'estimand',
        tier: estimand.tier,
        framework: estimand.framework,
        x: 150 + fwIndex * 300 + (offset % 2) * 80 - 40,
        y: tierY[estimand.tier as keyof typeof tierY] + Math.floor(offset / 2) * 15,
        fx: layoutMode === 'static' ? 150 + fwIndex * 300 + (offset % 2) * 80 - 40 : null,
        fy: layoutMode === 'static' ? tierY[estimand.tier as keyof typeof tierY] + Math.floor(offset / 2) * 15 : null
      });
      
      tierXOffset[estimand.framework][estimand.tier]++;
      
      // Link estimand to framework
      links.push({
        source: `fw_${estimand.framework}`,
        target: estimand.id,
        type: 'belongs_to'
      });
    });

    // Force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('x', d3.forceX((d: any) => d.x).strength(layoutMode === 'static' ? 1 : 0.2))
      .force('y', d3.forceY((d: any) => d.y).strength(layoutMode === 'static' ? 1 : 0.2))
      .force('collision', d3.forceCollide().radius(35))
      .force('charge', d3.forceManyBody().strength(layoutMode === 'dynamic' ? -300 : -50))
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance(100)
        .strength(layoutMode === 'dynamic' ? 0.4 : 0.1));

    // Add links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.4);

    // Add nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add circles
    node.append('circle')
      .attr('r', (d: any) => {
        if (d.type === 'theory') return 35;
        if (d.type === 'framework') return 28;
        return 18;
      })
      .attr('fill', (d: any) => {
        if (d.type === 'theory') return 'hsl(280 65% 60%)';
        if (d.type === 'framework') return 'hsl(215 70% 50%)';
        switch (d.tier) {
          case 'Basic': return 'hsl(142 70% 45%)';
          case 'Intermediate': return 'hsl(215 85% 55%)';
          case 'Advanced': return 'hsl(265 75% 58%)';
          case 'Frontier': return 'hsl(25 90% 55%)';
          default: return 'hsl(215 15% 60%)';
        }
      })
      .attr('stroke', 'white')
      .attr('stroke-width', 2.5)
      .attr('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))');

    // Add labels
    node.append('text')
      .text((d: any) => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', (d: any) => {
        if (d.type === 'theory') return 52;
        if (d.type === 'framework') return 45;
        return 32;
      })
      .attr('font-size', (d: any) => {
        if (d.type === 'theory') return '13px';
        if (d.type === 'framework') return '12px';
        return '10px';
      })
      .attr('font-weight', (d: any) => d.type !== 'estimand' ? 'bold' : 'normal')
      .attr('fill', 'hsl(215 25% 15%)')
      .style('pointer-events', 'none');

    // Click handlers
    node.on('click', (event: any, d: any) => {
      event.stopPropagation();
      if (d.type === 'estimand') {
        navigate(`/slides?id=${d.id}`);
      } else if (d.type === 'theory') {
        navigate(`/theory?id=${d.originalId}`);
      } else if (d.type === 'framework') {
        // Filter estimands page by framework
        navigate(`/estimands`);
      }
    });

    // Hover effects
    node.on('mouseenter', function() {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', (d: any) => {
          if (d.type === 'theory') return 40;
          if (d.type === 'framework') return 32;
          return 22;
        });
    }).on('mouseleave', function() {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', (d: any) => {
          if (d.type === 'theory') return 35;
          if (d.type === 'framework') return 28;
          return 18;
        });
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
      if (layoutMode === 'dynamic') {
        event.subject.fx = null;
        event.subject.fy = null;
      }
    }

    return () => {
      simulation.stop();
    };
  }, [layoutMode, navigate]);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        <div className="mb-6 flex flex-col md:flex-row items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold mb-2">Causal Inference Network</h1>
            <p className="text-muted-foreground max-w-2xl">
              Three-tier hierarchical view: Theory (top) → Frameworks (middle) → Estimands by Difficulty (bottom)
              <br />
              <span className="text-sm">Click any node to navigate to detailed slides and tutorials</span>
            </p>
          </div>
          <div className="flex gap-2">
            <Button 
              variant={layoutMode === 'static' ? 'default' : 'outline'}
              onClick={() => setLayoutMode('static')}
              size="sm"
            >
              Static Layout
            </Button>
            <Button 
              variant={layoutMode === 'dynamic' ? 'default' : 'outline'}
              onClick={() => setLayoutMode('dynamic')}
              size="sm"
            >
              Dynamic Layout
            </Button>
          </div>
        </div>

        {/* Legend */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <h3 className="font-semibold mb-3">Node Types & Difficulty Levels</h3>
          <div className="grid grid-cols-2 md:grid-cols-7 gap-3">
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-full" style={{ background: 'hsl(280 65% 60%)' }}></div>
              <span className="text-sm font-medium">Theory</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-full" style={{ background: 'hsl(215 70% 50%)' }}></div>
              <span className="text-sm font-medium">Framework</span>
            </div>
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
          </div>
        </div>

        {/* Network Visualization */}
        <div className="border rounded-lg bg-white p-4 shadow-sm">
          <svg ref={svgRef} className="w-full" style={{ height: '900px' }}></svg>
        </div>

        <div className="mt-4 p-4 rounded-lg border bg-muted/30">
          <p className="text-sm text-muted-foreground">
            <strong>How to use:</strong> {layoutMode === 'static' 
              ? 'Hierarchical layout with fixed tiers. Drag nodes to adjust positioning within tiers.'
              : 'Force-directed layout allows free exploration. Nodes cluster naturally based on connections. Drag to rearrange.'
            } Click theory nodes (purple) for conceptual slides, framework nodes (blue) for overview, or estimand nodes (colored by tier) for detailed tutorials.
          </p>
        </div>
      </div>
    </div>
  );
};

export default NetworkView;
