import Navigation from '@/components/Navigation';
import { useState } from 'react';
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import { BookOpen, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const TheoryLibrary = () => {
  const navigate = useNavigate();
  const [selectedTier, setSelectedTier] = useState<string>('all');
  
  const tiers = ['all', 'Foundational', 'Intermediate', 'Advanced', 'Frontier'];
  
  const filteredTopics = allTheoryTopics.filter(topic => {
    if (selectedTier !== 'all' && topic.tier !== selectedTier) return false;
    return true;
  });

  const getTierColor = (tier: string) => {
    const colors: Record<string, string> = {
      'Foundational': 'bg-blue-500/10 text-blue-700 border-blue-500/20',
      'Intermediate': 'bg-green-500/10 text-green-700 border-green-500/20',
      'Advanced': 'bg-orange-500/10 text-orange-700 border-orange-500/20',
      'Frontier': 'bg-purple-500/10 text-purple-700 border-purple-500/20'
    };
    return colors[tier] || 'bg-muted';
  };

  const handleTopicClick = (topicId: string) => {
    navigate(`/theory-overview?id=${topicId}`);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="bg-gradient-hero py-16">
        <div className="container">
          <div className="flex items-center gap-3 mb-4">
            <BookOpen className="h-10 w-10 text-primary-foreground" />
            <h1 className="text-4xl font-bold text-primary-foreground">
              Theory Library
            </h1>
          </div>
          <p className="text-xl text-primary-foreground/90 max-w-3xl">
            Browse and explore causal inference theory topics from mathematical foundations to frontier methods. Click any topic to view interactive slides.
          </p>
        </div>
      </section>

      {/* Theory Library */}
      <section className="py-12">
        <div className="container">
          {/* Filters */}
          <div className="mb-8 p-6 rounded-lg border bg-card">
            <h2 className="text-2xl font-bold mb-6">Filters</h2>
            
            <div className="space-y-4">
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
            </div>
            
            <p className="text-sm text-muted-foreground mt-4">
              Showing {filteredTopics.length} of {allTheoryTopics.length} topic{filteredTopics.length !== 1 ? 's' : ''}
            </p>
          </div>
          
          {/* Topics Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredTopics.map((topic) => (
              <Card 
                key={topic.id} 
                className="group hover:shadow-lg transition-all duration-300 cursor-pointer hover:scale-[1.02] border-2"
                onClick={() => handleTopicClick(topic.id)}
              >
                <CardHeader>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <Badge className={getTierColor(topic.tier)}>
                      {topic.tier}
                    </Badge>
                    <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                  </div>
                  <CardTitle className="text-xl group-hover:text-primary transition-colors">
                    {topic.title}
                  </CardTitle>
                  <CardDescription className="text-sm">
                    {topic.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {/* Prerequisites */}
                    {topic.prerequisites && topic.prerequisites.length > 0 && (
                      <div>
                        <p className="text-xs font-medium text-muted-foreground mb-1">Prerequisites:</p>
                        <div className="flex flex-wrap gap-1">
                          {topic.prerequisites.slice(0, 2).map((prereq, idx) => (
                            <Badge key={idx} variant="secondary" className="text-xs">
                              {allTheoryTopics.find(t => t.id === prereq)?.title.slice(0, 20) || prereq}
                            </Badge>
                          ))}
                          {topic.prerequisites.length > 2 && (
                            <Badge variant="secondary" className="text-xs">
                              +{topic.prerequisites.length - 2} more
                            </Badge>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Learning Objectives Count */}
                    {topic.learningObjectives && (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <BookOpen className="h-3 w-3" />
                        <span>{topic.learningObjectives.length} learning objectives</span>
                      </div>
                    )}

                    {/* View Overview Button */}
                    <Button 
                      variant="outline" 
                      className="w-full mt-2 group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                      size="sm"
                    >
                      View Learning Path
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          {filteredTopics.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              No topics found for the selected filters
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default TheoryLibrary;
