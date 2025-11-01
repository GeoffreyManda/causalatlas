import Navigation from '@/components/Navigation';
import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { estimandsData } from '@/data/estimands';
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { estimandFamilies } from '@/data/estimandFamilies';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { useNavigate, useSearchParams } from 'react-router-dom';

const NetworkView = () => {
  const estimandsSvgRef = useRef<SVGSVGElement>(null);
  const theorySvgRef = useRef<SVGSVGElement>(null);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const highlightNodeId = searchParams.get('node');
  const [activeTab, setActiveTab] = useState<string>('estimands');
  
  // State for estimands filters
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [selectedDesign, setSelectedDesign] = useState<string>('all');
  const [selectedFamily, setSelectedFamily] = useState<string>('all');
  
  // State for theory filters
  const [selectedTheoryTier, setSelectedTheoryTier] = useState<string>('all');

  // Calculate available filter options for estimands
  const getFilteredEstimands = () => {
    return estimandsData.filter(e => {
      if (selectedTier !== 'all' && e.tier !== selectedTier) return false;
      if (selectedFramework !== 'all' && e.framework !== selectedFramework) return false;
      if (selectedDesign !== 'all' && e.design !== selectedDesign) return false;
      if (selectedFamily !== 'all' && e.estimand_family !== selectedFamily) return false;
      return true;
    });
  };

  const currentFiltered = getFilteredEstimands();
  
  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(currentFiltered.map(e => e.framework)))];
  const designs = ['all', ...Array.from(new Set(currentFiltered.map(e => e.design))).sort()];
  const families = ['all', ...Array.from(new Set(currentFiltered.map(e => e.estimand_family))).sort()];

  // Estimands Network Effect
  useEffect(() => {
    if (!estimandsSvgRef.current || activeTab !== 'estimands') return;

    const width = 1800;
    const height = 1200;

    d3.select(estimandsSvgRef.current).selectAll('*').remove();

    const svg = d3.select(estimandsSvgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('class', 'w-full h-full');

    const filteredEstimands = currentFiltered;

    // Build hierarchy: Root → Framework → Design → Family → Estimands
    const hierarchy: any = { name: 'Causal Inference', type: 'root', children: [] };
    const frameworkMap = new Map();

    filteredEstimands.forEach(est => {
      if (!frameworkMap.has(est.framework)) {
        const fwNode = { name: est.framework, type: 'framework', children: [] };
        frameworkMap.set(est.framework, fwNode);
        hierarchy.children.push(fwNode);
      }
      const fwNode = frameworkMap.get(est.framework);

      let designNode = fwNode.children.find((d: any) => d.name === est.design);
      if (!designNode) {
        designNode = { name: est.design, type: 'design', children: [] };
        fwNode.children.push(designNode);
      }

      let familyNode = designNode.children.find((f: any) => f.name === est.estimand_family);
      if (!familyNode) {
        familyNode = { name: est.estimand_family, type: 'family', children: [] };
        designNode.children.push(familyNode);
      }

      familyNode.children.push({
        name: est.short_name,
        type: 'estimand',
        tier: est.tier,
        id: est.id,
        size: 1
      });
    });

    const root = d3.hierarchy(hierarchy)
      .sum((d: any) => d.size || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const treeLayout = d3.tree<any>()
      .size([height - 100, width - 400])
      .separation((a, b) => (a.parent === b.parent ? 1 : 1.2));

    treeLayout(root);

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

    const node = svg.append('g')
      .selectAll('g')
      .data(root.descendants())
      .join('g')
      .attr('transform', (d: any) => `translate(${d.y + 200},${d.x + 50})`)
      .style('cursor', 'pointer');

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

    node.on('click', (event: any, d: any) => {
      event.stopPropagation();
      if (d.data.type === 'estimand') {
        navigate(`/estimand-overview?id=${d.data.id}`);
      }
    });

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
      
      circle.transition().duration(200)
        .attr('r', originalRadius)
        .attr('stroke-width', 2);
    });

  }, [selectedTier, selectedFramework, selectedDesign, selectedFamily, highlightNodeId, activeTab, navigate, currentFiltered]);

  // Theory Network Effect
  useEffect(() => {
    if (!theorySvgRef.current || activeTab !== 'theory') return;

    const filteredTopics = allTheoryTopics.filter(topic => {
      if (selectedTheoryTier !== 'all' && topic.tier !== selectedTheoryTier) return false;
      return true;
    });

    const svg = d3.select(theorySvgRef.current);
    svg.selectAll('*').remove();

    const width = 1200;
    const height = 800;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    // Build hierarchy: root → tiers → topics
    const groupedByTier = d3.group(filteredTopics, d => d.tier);
    const hierarchyData: any = {
      name: 'Theory Topics',
      children: Array.from(groupedByTier, ([tier, topics]) => ({
        name: tier,
        children: topics.map(t => ({ name: t.title, id: t.id, tier: t.tier }))
      }))
    };

    const root = d3.hierarchy(hierarchyData)
      .sum(() => 1)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    d3.treemap<any>()
      .size([width - margin.left - margin.right, height - margin.top - margin.bottom])
      .padding(2)
      (root);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const tierColors: Record<string, string> = {
      'Foundational': '#3b82f6',
      'Intermediate': '#10b981',
      'Advanced': '#f59e0b',
      'Frontier': '#a855f7'
    };

    const cell = g.selectAll('g')
      .data(root.leaves())
      .enter()
      .append('g')
      .attr('transform', (d: any) => `translate(${d.x0},${d.y0})`);

    cell.append('rect')
      .attr('width', (d: any) => d.x1 - d.x0)
      .attr('height', (d: any) => d.y1 - d.y0)
      .attr('fill', (d: any) => tierColors[d.data.tier] || '#6b7280')
      .attr('opacity', 0.7)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function() {
        d3.select(this).attr('opacity', 0.9);
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.7);
      })
      .on('click', (event: any, d: any) => {
        navigate(`/theory-overview?id=${d.data.id}`);
      });

    cell.append('text')
      .attr('x', 4)
      .attr('y', 16)
      .text((d: any) => d.data.name)
      .attr('font-size', '11px')
      .attr('fill', '#fff')
      .attr('font-weight', 'bold')
      .style('pointer-events', 'none')
      .each(function(d: any) {
        const self = d3.select(this);
        const width = d.x1 - d.x0 - 8;
        let text = d.data.name;
        self.text(text);
        while (self.node()!.getComputedTextLength() > width && text.length > 0) {
          text = text.slice(0, -1);
          self.text(text + '...');
        }
      });

  }, [selectedTheoryTier, activeTab, navigate]);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-16">
        <div className="container">
          <h1 className="text-4xl font-bold text-primary-foreground mb-4">
            Interactive Network View
          </h1>
          <p className="text-xl text-primary-foreground/90 max-w-3xl">
            Explore causal inference concepts as interactive hierarchical networks. Switch between estimands and theory topics.
          </p>
        </div>
      </section>

      {/* Network Visualization */}
      <section className="py-12">
        <div className="container">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 mb-8">
              <TabsTrigger value="estimands">Estimands Network</TabsTrigger>
              <TabsTrigger value="theory">Theory Network</TabsTrigger>
            </TabsList>

            {/* ESTIMANDS TAB */}
            <TabsContent value="estimands">
              {/* Estimands Filters */}
              <div className="mb-8 p-6 rounded-lg border bg-card">
                <h2 className="text-2xl font-bold mb-6">Filters</h2>
                <div className="space-y-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h3 className="text-sm font-medium mb-3">Filter by Tier</h3>
                      <div className="flex flex-wrap gap-2">
                        {tiers.map((tier) => (
                          <Badge
                            key={tier}
                            variant={selectedTier === tier ? "default" : "outline"}
                            className="cursor-pointer px-4 py-2 text-sm hover:scale-105 transition-transform"
                            onClick={() => setSelectedTier(tier)}
                          >
                            {tier === 'all' ? 'All Tiers' : tier}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium mb-3">Filter by Family</h3>
                      <div className="flex flex-wrap gap-2">
                        {families.map((family) => (
                          <Badge
                            key={family}
                            variant={selectedFamily === family ? "default" : "outline"}
                            className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                            onClick={() => setSelectedFamily(family)}
                          >
                            {family === 'all' ? 'All Families' : family}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h3 className="text-sm font-medium mb-3">Filter by Framework</h3>
                      <div className="flex flex-wrap gap-2">
                        {frameworks.map((fw) => (
                          <Badge
                            key={fw}
                            variant={selectedFramework === fw ? "default" : "outline"}
                            className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                            onClick={() => setSelectedFramework(fw)}
                          >
                            {fw === 'all' ? 'All Frameworks' : fw}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium mb-3">Filter by Study Design</h3>
                      <div className="flex flex-wrap gap-2">
                        {designs.map((design) => (
                          <Badge
                            key={design}
                            variant={selectedDesign === design ? "default" : "outline"}
                            className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                            onClick={() => setSelectedDesign(design)}
                          >
                            {design === 'all' ? 'All Designs' : design.replace(/_/g, ' ')}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Estimands Legend */}
              <Card className="mb-6 p-4">
                <h3 className="font-semibold mb-3">Legend</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-[hsl(280_85%_55%)]"></div>
                    <span className="text-sm">Root</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-[hsl(215_70%_50%)]"></div>
                    <span className="text-sm">Framework</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-[hsl(195_75%_40%)]"></div>
                    <span className="text-sm">Design</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-[hsl(265_60%_45%)]"></div>
                    <span className="text-sm">Family</span>
                  </div>
                </div>
              </Card>

              <div className="overflow-x-auto">
                <svg ref={estimandsSvgRef} width="1800" height="1200" className="mx-auto border rounded-lg"></svg>
              </div>
            </TabsContent>

            {/* THEORY TAB */}
            <TabsContent value="theory">
              {/* Theory Filters */}
              <div className="mb-8 p-6 rounded-lg border bg-card">
                <h2 className="text-2xl font-bold mb-6">Filters</h2>
                <div>
                  <h3 className="text-sm font-medium mb-3">Filter by Tier</h3>
                  <div className="flex flex-wrap gap-2">
                    {['all', 'Foundational', 'Intermediate', 'Advanced', 'Frontier'].map((tier) => (
                      <Badge
                        key={tier}
                        variant={selectedTheoryTier === tier ? "default" : "outline"}
                        className="cursor-pointer px-4 py-2 text-sm hover:scale-105 transition-transform"
                        onClick={() => setSelectedTheoryTier(tier)}
                      >
                        {tier === 'all' ? 'All Tiers' : tier}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              {/* Theory Legend */}
              <Card className="mb-6 p-4">
                <h3 className="font-semibold mb-3">Legend</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-[#3b82f6]"></div>
                    <span className="text-sm">Foundational</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-[#10b981]"></div>
                    <span className="text-sm">Intermediate</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-[#f59e0b]"></div>
                    <span className="text-sm">Advanced</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-[#a855f7]"></div>
                    <span className="text-sm">Frontier</span>
                  </div>
                </div>
              </Card>

              <div className="overflow-x-auto">
                <svg ref={theorySvgRef} width="1200" height="800" className="mx-auto border rounded-lg"></svg>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </section>
    </div>
  );
};

export default NetworkView;
