import { useEffect, useRef } from 'react';
import Navigation from '@/components/Navigation';
import { estimandsData } from '@/data/estimands';
import * as d3 from 'd3';

const NetworkView = () => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 1200;
    const height = 800;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('class', 'w-full h-full');

    // Create nodes and links
    const nodes: any[] = [];
    const links: any[] = [];

    // Add framework nodes
    const frameworks = Array.from(new Set(estimandsData.map(e => e.framework)));
    frameworks.forEach((fw, i) => {
      nodes.push({ id: `fw_${fw}`, label: fw, type: 'framework', group: 0 });
    });

    // Add estimand nodes
    estimandsData.forEach((estimand, i) => {
      nodes.push({
        id: estimand.id,
        label: estimand.short_name,
        type: 'estimand',
        tier: estimand.tier,
        group: 1 + frameworks.indexOf(estimand.framework),
      });
      
      // Link to framework
      links.push({
        source: `fw_${estimand.framework}`,
        target: estimand.id,
        type: 'implements',
      });
    });

    // Create simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(50));

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
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Estimand Network</h1>
          <p className="text-muted-foreground">
            Interactive visualization of causal estimands and their relationships
          </p>
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
          Drag nodes to explore • Larger nodes represent frameworks • Colors indicate complexity tier
        </p>
      </div>
    </div>
  );
};

export default NetworkView;
