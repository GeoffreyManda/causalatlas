import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import { estimandsData } from '@/data/estimands';
import { causalTheory } from '@/data/theory';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import * as d3 from 'd3';

const NetworkView = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const navigate = useNavigate();
  const [layoutMode, setLayoutMode] = useState<'static' | 'dynamic'>('static');
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [highlightedType, setHighlightedType] = useState<string | null>(null);

  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(estimandsData.map(e => e.framework)))];

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 1600;
    const height = 1000;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('class', 'w-full h-full');

    const nodes: any[] = [];
    const links: any[] = [];

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

    const allFrameworks = Array.from(new Set(estimandsData.map(e => e.framework)));
    const filteredFrameworksSet = new Set(
      estimandsData
        .filter(e => selectedTier === 'all' || e.tier === selectedTier)
        .map(e => e.framework)
    );
    const frameworksToShow = selectedFramework === 'all' 
      ? allFrameworks.filter(fw => filteredFrameworksSet.has(fw))
      : allFrameworks.filter(fw => fw === selectedFramework);
    
    frameworksToShow.forEach((fw, i) => {
      nodes.push({
        id: `fw_${fw}`,
        label: fw.replace(/([A-Z])/g, ' $1').trim(),
        type: 'framework',
        framework: fw,
        x: 150 + i * 300,
        y: 280,
        fx: layoutMode === 'static' ? 150 + i * 300 : null,
        fy: layoutMode === 'static' ? 280 : null
      });
    });

    const tierY = { Basic: 500, Intermediate: 650, Advanced: 800, Frontier: 920 };
    const tierXOffset: any = {};
    
    frameworksToShow.forEach(fw => {
      tierXOffset[fw] = { Basic: 0, Intermediate: 0, Advanced: 0, Frontier: 0 };
    });

    const filteredEstimands = estimandsData.filter(e => {
      if (selectedTier !== 'all' && e.tier !== selectedTier) return false;
      if (selectedFramework !== 'all' && e.framework !== selectedFramework) return false;
      return true;
    });

    filteredEstimands.forEach((estimand) => {
      const fwIndex = frameworksToShow.indexOf(estimand.framework);
      if (fwIndex === -1) return;
      
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
      
      links.push({
        source: `fw_${estimand.framework}`,
        target: estimand.id,
        type: 'belongs_to'
      });
    });

    const simulation = d3.forceSimulation(nodes)
      .force('x', d3.forceX((d: any) => d.x).strength(layoutMode === 'static' ? 1 : 0.2))
      .force('y', d3.forceY((d: any) => d.y).strength(layoutMode === 'static' ? 1 : 0.2))
      .force('collision', d3.forceCollide().radius(35))
      .force('charge', d3.forceManyBody().strength(layoutMode === 'dynamic' ? -300 : -50))
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance(100)
        .strength(layoutMode === 'dynamic' ? 0.4 : 0.1));

    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.4);

    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('circle')
      .attr('r', (d: any) => d.type === 'theory' ? 35 : d.type === 'framework' ? 28 : 18)
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
      .attr('opacity', (d: any) => {
        if (!highlightedType) return 1;
        if (d.type === highlightedType || d.tier === highlightedType) return 1;
        return 0.2;
      })
      .attr('stroke', 'white')
      .attr('stroke-width', 2.5)
      .attr('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))');

    node.append('text')
      .text((d: any) => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', (d: any) => d.type === 'theory' ? 52 : d.type === 'framework' ? 45 : 32)
      .attr('font-size', (d: any) => d.type === 'theory' ? '13px' : d.type === 'framework' ? '12px' : '10px')
      .attr('font-weight', (d: any) => d.type !== 'estimand' ? 'bold' : 'normal')
      .attr('fill', 'hsl(215 25% 15%)')
      .style('pointer-events', 'none');

    node.on('click', (event: any, d: any) => {
      event.stopPropagation();
      if (d.type === 'estimand') {
        navigate(`/slides?id=${d.id}`, { state: { from: '/network' } });
      } else if (d.type === 'theory') {
        navigate(`/theory?id=${d.originalId}`, { state: { from: '/network' } });
      } else if (d.type === 'framework') {
        setSelectedFramework(d.framework);
      }
    });

    node.on('mouseenter', function() {
      d3.select(this).select('circle').transition().duration(200)
        .attr('r', (d: any) => d.type === 'theory' ? 40 : d.type === 'framework' ? 32 : 22);
    }).on('mouseleave', function() {
      d3.select(this).select('circle').transition().duration(200)
        .attr('r', (d: any) => d.type === 'theory' ? 35 : d.type === 'framework' ? 28 : 18);
    });

    simulation.on('tick', () => {
      link.attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y);
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

    return () => simulation.stop();
  }, [layoutMode, navigate, selectedTier, selectedFramework, highlightedType]);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Causal Inference Network</h1>
          <p className="text-muted-foreground max-w-3xl">
            Three-tier hierarchical view: Theory → Frameworks → Estimands. Click nodes to navigate or use filters.
          </p>
        </div>

        <div className="mb-6 flex justify-end gap-2">
          <Button variant={layoutMode === 'static' ? 'default' : 'outline'} onClick={() => setLayoutMode('static')} size="sm">Static</Button>
          <Button variant={layoutMode === 'dynamic' ? 'default' : 'outline'} onClick={() => setLayoutMode('dynamic')} size="sm">Dynamic</Button>
        </div>

        <div className="mb-6 p-4 rounded-lg border bg-card">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Filter Tier</label>
              <div className="flex flex-wrap gap-2">
                {tiers.map(tier => (
                  <Badge key={tier} variant={selectedTier === tier ? 'default' : 'outline'}
                    className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                    onClick={() => setSelectedTier(tier)}>
                    {tier === 'all' ? 'All' : tier}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Filter Framework</label>
              <div className="flex flex-wrap gap-2">
                {frameworks.map(fw => (
                  <Badge key={fw} variant={selectedFramework === fw ? 'default' : 'outline'}
                    className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                    onClick={() => setSelectedFramework(fw)}>
                    {fw === 'all' ? 'All' : fw.replace(/([A-Z])/g, ' $1').trim()}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>

        <Card className="mb-6 p-4">
          <h3 className="font-semibold mb-3">Legend (Click to Highlight)</h3>
          <div className="grid grid-cols-2 md:grid-cols-7 gap-3">
            {[
              { type: 'theory', color: 'hsl(280 65% 60%)', label: 'Theory' },
              { type: 'framework', color: 'hsl(215 70% 50%)', label: 'Framework' },
              { type: 'Basic', color: null, label: 'Basic' },
              { type: 'Intermediate', color: null, label: 'Intermediate' },
              { type: 'Advanced', color: null, label: 'Advanced' },
              { type: 'Frontier', color: null, label: 'Frontier' }
            ].map(item => (
              <button key={item.type}
                onClick={() => setHighlightedType(highlightedType === item.type ? null : item.type)}
                className={`flex items-center gap-2 p-2 rounded hover:bg-muted/50 transition-colors ${highlightedType === item.type ? 'bg-muted' : ''}`}>
                <div className={`h-6 w-6 rounded-full ${item.color ? '' : `bg-tier-${item.type.toLowerCase()}`}`} 
                  style={item.color ? { background: item.color } : undefined}></div>
                <span className="text-sm">{item.label}</span>
              </button>
            ))}
          </div>
        </Card>

        <div className="border rounded-lg bg-card p-4 shadow-sm">
          <svg ref={svgRef} className="w-full" style={{ height: '900px' }}></svg>
        </div>
      </div>
    </div>
  );
};

export default NetworkView;
