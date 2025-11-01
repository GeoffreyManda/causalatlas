import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface GraphicsCanvasProps {
  code?: string;
  language?: string;
}

export const GraphicsCanvas = ({ code, language }: GraphicsCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!code) return;

    // Clear previous renders
    if (svgRef.current) {
      d3.select(svgRef.current).selectAll('*').remove();
    }

    try {
      // Example: Parse simple plot commands
      if (code.includes('plot') || code.includes('scatter')) {
        renderSampleChart();
      }
    } catch (error) {
      console.error('Graphics render error:', error);
    }
  }, [code]);

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
      </div>
    </div>
  );
};
