import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import TheorySlide from '@/components/TheorySlide';
import { causalTheory } from '@/data/theory';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, BookOpen } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

const TheoryView = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const topicId = searchParams.get('id') || causalTheory[0].id;
  const [slideIndex, setSlideIndex] = useState(0);

  const currentTopic = causalTheory.find(t => t.id === topicId) || causalTheory[0];
  
  // Calculate total slides dynamically based on content
  const contentParagraphs = currentTopic.content.split('\n\n').filter(p => p.trim()).length;
  const objectiveSlides = Math.ceil(currentTopic.learningObjectives.length / 3);
  const definitionSlides = Math.ceil(currentTopic.keyDefinitions.length / 3);
  const contentSlides = Math.ceil(contentParagraphs / 3);
  const referenceSlides = Math.ceil(currentTopic.references.length / 3);
  const totalSlides = 1 + objectiveSlides + definitionSlides + contentSlides + 2 + referenceSlides; // title + objectives + definitions + content + 2 code + references

  // Reset slide index when topic changes
  useEffect(() => {
    setSlideIndex(0);
  }, [topicId]);

  const goToNext = () => {
    if (slideIndex < totalSlides - 1) {
      setSlideIndex(slideIndex + 1);
    } else {
      // Move to next topic
      const currentIdx = causalTheory.findIndex(t => t.id === topicId);
      if (currentIdx < causalTheory.length - 1) {
        setSearchParams({ id: causalTheory[currentIdx + 1].id });
        setSlideIndex(0);
      }
    }
  };

  const goToPrevious = () => {
    if (slideIndex > 0) {
      setSlideIndex(slideIndex - 1);
    } else {
      // Move to previous topic
      const currentIdx = causalTheory.findIndex(t => t.id === topicId);
      if (currentIdx > 0) {
        setSearchParams({ id: causalTheory[currentIdx - 1].id });
        setSlideIndex(totalSlides - 1);
      }
    }
  };

  const handleTopicChange = (newTopicId: string) => {
    setSearchParams({ id: newTopicId });
    setSlideIndex(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <Navigation />
      
      <div className="container mx-auto px-4 py-8">
        {/* Header with topic selector */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <BookOpen className="h-6 w-6 text-primary" />
            <Select value={topicId} onValueChange={handleTopicChange}>
              <SelectTrigger className="w-[400px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {causalTheory.map(topic => (
                  <SelectItem key={topic.id} value={topic.id}>
                    {topic.title}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="text-sm text-muted-foreground">
            Slide {slideIndex + 1} of {totalSlides}
          </div>
        </div>

        {/* Slide */}
        <div className="flex items-center justify-center mb-8">
          <TheorySlide topic={currentTopic} slideIndex={slideIndex} />
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={goToPrevious}
            disabled={slideIndex === 0 && causalTheory.findIndex(t => t.id === topicId) === 0}
            className="gap-2"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>

          <div className="flex gap-2">
            {Array.from({ length: totalSlides }).map((_, idx) => (
              <button
                key={idx}
                onClick={() => setSlideIndex(idx)}
                className={`w-2 h-2 rounded-full transition-all ${
                  idx === slideIndex 
                    ? 'bg-primary w-8' 
                    : 'bg-muted-foreground/30 hover:bg-muted-foreground/50'
                }`}
                aria-label={`Go to slide ${idx + 1}`}
              />
            ))}
          </div>

          <Button
            variant="outline"
            onClick={goToNext}
            disabled={slideIndex === totalSlides - 1 && causalTheory.findIndex(t => t.id === topicId) === causalTheory.length - 1}
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
