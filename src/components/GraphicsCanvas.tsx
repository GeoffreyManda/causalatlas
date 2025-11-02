import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface GraphicsCanvasProps {
  code?: string;
  language?: string;
}

interface DAGNode {
  id: string;
  label: string;
  x?: number;
  y?: number;
}

interface DAGEdge {
  source: string;
  target: string;
  type?: 'causal' | 'confounding' | 'selection';
}

export const GraphicsCanvas = ({ code, language }: GraphicsCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dagData, setDagData] = useState<{ nodes: DAGNode[]; edges: DAGEdge[] } | null>(null);

  useEffect(() => {
    if (!code) return;

    // Clear previous renders
    if (svgRef.current) {
      d3.select(svgRef.current).selectAll('*').remove();
    }

    try {
      // Check if code generates a DAG
      if (code.includes('dag(') || code.includes('DAG(') || code.includes('create_dag')) {
        const dag = parseDAGFromCode(code);
        if (dag) {
          setDagData(dag);
          renderDAG(dag);
          return;
        }
      }
      
      // Otherwise render sample chart
      if (code.includes('plot') || code.includes('scatter')) {
        setDagData(null);
        renderSampleChart();
      }
    } catch (error) {
      console.error('Graphics render error:', error);
    }
  }, [code]);

  const parseDAGFromCode = (code: string): { nodes: DAGNode[]; edges: DAGEdge[] } | null => {
    try {
      // Look for dag function calls with nodes and edges
      const dagMatch = code.match(/dag\s*\(\s*nodes\s*=\s*\[(.*?)\]\s*,\s*edges\s*=\s*\[(.*?)\]/s);
      if (!dagMatch) return null;

      const nodesStr = dagMatch[1];
      const edgesStr = dagMatch[2];

      // Parse nodes
      const nodeMatches = nodesStr.matchAll(/['"]([^'"]+)['"]/g);
      const nodes: DAGNode[] = Array.from(nodeMatches).map(m => ({
        id: m[1],
        label: m[1]
      }));

      // Parse edges
      const edgeMatches = edgesStr.matchAll(/\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)/g);
      const edges: DAGEdge[] = Array.from(edgeMatches).map(m => ({
        source: m[1],
        target: m[2],
        type: 'causal'
      }));

      return { nodes, edges };
    } catch (e) {
      return null;
    }
  };

  const renderDAG = (data: { nodes: DAGNode[]; edges: DAGEdge[] }) => {
    if (!svgRef.current) return;

    const width = 700;
    const height = 500;
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create force simulation for layout
    const simulation = d3.forceSimulation(data.nodes as any)
      .force('link', d3.forceLink(data.edges as any)
        .id((d: any) => d.id)
        .distance(150))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(50));

    // Add arrow markers
    svg.append('defs').selectAll('marker')
      .data(['causal', 'confounding', 'selection'])
      .enter().append('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => d === 'causal' ? 'hsl(var(--primary))' : 
                         d === 'confounding' ? 'hsl(var(--destructive))' : 
                         'hsl(var(--secondary))');

    // Draw edges
    const link = svg.append('g')
      .selectAll('line')
      .data(data.edges)
      .enter().append('line')
      .attr('stroke', d => d.type === 'causal' ? 'hsl(var(--primary))' : 
                           d.type === 'confounding' ? 'hsl(var(--destructive))' : 
                           'hsl(var(--secondary))')
      .attr('stroke-width', 2)
      .attr('marker-end', d => `url(#arrow-${d.type || 'causal'})`);

    // Draw nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(data.nodes)
      .enter().append('g')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    node.append('circle')
      .attr('r', 20)
      .attr('fill', 'hsl(var(--card))')
      .attr('stroke', 'hsl(var(--primary))')
      .attr('stroke-width', 2);

    node.append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('fill', 'hsl(var(--foreground))')
      .style('font-weight', 'bold')
      .style('font-size', '12px')
      .style('pointer-events', 'none');

    // Update positions on simulation tick
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
  };

  const renderSampleChart = () => {
    if (!svgRef.current) return;

    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Sample data
    const data = Array.from({ length: 20 }, (_, i) => ({
      x: i,
      y: Math.random() * 100 + Math.sin(i / 3) * 50
    }));

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.x) || 0])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.y) || 0])
      .range([height - margin.bottom, margin.top]);

    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale));

    // Line
    const line = d3.line<{ x: number; y: number }>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y));

    svg.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', 'hsl(var(--primary))')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Points
    svg.selectAll('circle')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 4)
      .attr('fill', 'hsl(var(--primary))');
  };

  return (
    <div className="h-full bg-background p-4 overflow-auto">
      <div className="flex flex-col items-center justify-center min-h-full">
        <canvas 
          ref={canvasRef} 
          className="hidden border border-border rounded-lg"
          width={600}
          height={400}
        />
        <svg 
          ref={svgRef} 
          className="border border-border rounded-lg bg-card"
        />
        {!code && (
          <div className="text-center text-muted-foreground mt-8">
            <p className="text-lg font-medium mb-2">Graphics Visualization</p>
            <p className="text-sm">Run code with plot/visualization commands to see output here</p>
          </div>
        )}
        {dagData && (
          <div className="mt-4 p-3 bg-muted rounded-lg text-sm">
            <p className="font-semibold mb-1">DAG Structure:</p>
            <p className="text-muted-foreground">
              {dagData.nodes.length} nodes, {dagData.edges.length} edges
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Drag nodes to rearrange • Arrows show causal direction
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
