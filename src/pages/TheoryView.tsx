import { useState } from 'react';
import Navigation from '@/components/Navigation';
import { causalTheory, TheoryTopic } from '@/data/theory';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, BookOpen } from 'lucide-react';

const TheoryView = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const currentTopic = causalTheory[currentIndex];

  const goToNext = () => {
    if (currentIndex < causalTheory.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goToPrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const getTierColor = (tier: string) => {
    const colors = {
      Foundational: 'bg-green-600',
      Intermediate: 'bg-blue-600',
      Advanced: 'bg-purple-600',
    };
    return colors[tier as keyof typeof colors] || 'bg-gray-600';
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container py-8">
        {/* Progress */}
        <div className="mb-6 flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Topic {currentIndex + 1} of {causalTheory.length}
          </div>
          <Badge className={getTierColor(currentTopic.tier)}>
            {currentTopic.tier}
          </Badge>
        </div>

        {/* Main Content */}
        <Card className="mb-8">
          <CardHeader>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-primary/10 rounded-lg">
                <BookOpen className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-3xl mb-2">{currentTopic.title}</CardTitle>
                <p className="text-muted-foreground">{currentTopic.description}</p>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-6">
            {/* Learning Objectives */}
            <div>
              <h3 className="font-semibold text-lg mb-3">Learning Objectives</h3>
              <ul className="space-y-2">
                {currentTopic.learningObjectives.map((obj, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-primary mt-1">✓</span>
                    <span>{obj}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Prerequisites */}
            {currentTopic.prerequisites.length > 0 && (
              <div className="p-4 bg-muted/50 rounded-lg">
                <h4 className="font-semibold mb-2">Prerequisites</h4>
                <div className="flex flex-wrap gap-2">
                  {currentTopic.prerequisites.map((prereq, idx) => {
                    const prereqTopic = causalTheory.find(t => t.id === prereq);
                    return (
                      <Badge key={idx} variant="outline">
                        {prereqTopic?.title || prereq}
                      </Badge>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Content */}
            <div className="prose prose-slate dark:prose-invert max-w-none">
              <div className="whitespace-pre-wrap leading-relaxed">
                {currentTopic.content}
              </div>
            </div>

            {/* Key Definitions */}
            <div>
              <h3 className="font-semibold text-lg mb-3">Key Definitions</h3>
              <div className="space-y-3">
                {currentTopic.keyDefinitions.map((def, idx) => (
                  <div key={idx} className="p-4 bg-muted/30 rounded-lg">
                    <div className="font-semibold text-primary mb-1">{def.term}</div>
                    <div className="text-sm">{def.definition}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Code Examples */}
            <div>
              <h3 className="font-semibold text-lg mb-3">Interactive Example</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <Badge className="mb-2">Python</Badge>
                  <pre className="text-xs bg-terminal-bg text-terminal-fg p-4 rounded-lg overflow-x-auto">
                    <code>{currentTopic.examples.python}</code>
                  </pre>
                </div>
                <div>
                  <Badge className="mb-2">R</Badge>
                  <pre className="text-xs bg-terminal-bg text-terminal-fg p-4 rounded-lg overflow-x-auto">
                    <code>{currentTopic.examples.r}</code>
                  </pre>
                </div>
              </div>
            </div>

            {/* References */}
            <div>
              <h3 className="font-semibold text-lg mb-3">References</h3>
              <ul className="space-y-2 text-sm">
                {currentTopic.references.map((ref, idx) => (
                  <li key={idx} className="text-muted-foreground">
                    {ref.authors} ({ref.year}). <em>{ref.title}</em>. DOI: {ref.doi}
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={goToPrevious}
            disabled={currentIndex === 0}
            className="gap-2"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>

          <div className="flex gap-2">
            {causalTheory.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentIndex(idx)}
                className={`w-2 h-2 rounded-full transition-all ${
                  idx === currentIndex 
                    ? 'bg-primary w-8' 
                    : 'bg-muted-foreground/30 hover:bg-muted-foreground/50'
                }`}
                aria-label={`Go to topic ${idx + 1}`}
              />
            ))}
          </div>

          <Button
            variant="outline"
            onClick={goToNext}
            disabled={currentIndex === causalTheory.length - 1}
            className="gap-2"
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default TheoryView;
