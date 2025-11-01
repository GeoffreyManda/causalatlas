import { useState, useEffect } from 'react';
import { useSearchParams, useLocation, useNavigate } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import TheorySlide from '@/components/TheorySlide';
import { causalTheory } from '@/data/theory';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, BookOpen, Download, ArrowLeft } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

const TheoryView = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const topicId = searchParams.get('id') || causalTheory[0].id;
  const [slideIndex, setSlideIndex] = useState(0);
  const [selectedTier, setSelectedTier] = useState<string>('all');

  // Get referrer from state or default to learning hub
  const referrer = (location.state as any)?.from || '/learning';

  const currentTopic = causalTheory.find(t => t.id === topicId) || causalTheory[0];
  
  // Calculate total slides dynamically based on content
  const contentParagraphs = currentTopic.content.split('\n\n').filter(p => p.trim()).length;
  const objectiveSlides = Math.ceil(currentTopic.learningObjectives.length / 3);
  const definitionSlides = Math.ceil(currentTopic.keyDefinitions.length / 3);
  const contentSlides = Math.ceil(contentParagraphs / 3);
  const referenceSlides = Math.ceil(currentTopic.references.length / 3);
  const totalSlides = 1 + objectiveSlides + definitionSlides + contentSlides + 2 + referenceSlides; // title + objectives + definitions + content + 2 code + references

  // Filter topics
  const filteredTopics = causalTheory.filter(t => {
    if (selectedTier !== 'all' && t.tier !== selectedTier) return false;
    return true;
  });

  const tiers = ['all', 'Foundational', 'Intermediate', 'Advanced'];

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

  const downloadSlides = () => {
    toast.info('Theory slide download coming soon (paywall feature)');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <Navigation />
      
      <div className="container mx-auto px-4 py-8">
        {/* Back button and download */}
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            onClick={() => navigate(referrer)}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <Button
            variant="outline"
            onClick={downloadSlides}
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            Download Slides (Premium)
          </Button>
        </div>

        {/* Filter */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <label className="text-sm font-medium mb-2 block">Filter by Level</label>
          <div className="flex flex-wrap gap-2">
            {tiers.map(tier => (
              <Badge
                key={tier}
                variant={selectedTier === tier ? 'default' : 'outline'}
                className="cursor-pointer px-4 py-2 hover:scale-105 transition-transform"
                onClick={() => setSelectedTier(tier)}
              >
                {tier === 'all' ? 'All Levels' : tier}
              </Badge>
            ))}
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Showing {filteredTopics.length} of {causalTheory.length} topics
          </p>
        </div>

        {/* Header with topic selector */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <BookOpen className="h-6 w-6 text-primary" />
            <Select value={topicId} onValueChange={handleTopicChange}>
              <SelectTrigger className="w-full max-w-[400px] bg-popover">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[400px] bg-popover z-50">
                {filteredTopics.map(topic => (
                  <SelectItem key={topic.id} value={topic.id}>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">{topic.tier}</Badge>
                      {topic.title}
                    </div>
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
