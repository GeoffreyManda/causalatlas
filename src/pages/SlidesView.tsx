import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { estimandsData } from '@/data/estimands';
import EstimandSlideStandalone from '@/components/EstimandSlideStandalone';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Presentation } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

const SlidesView = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const estimandId = searchParams.get('id') || estimandsData[0].id;
  const [slideIndex, setSlideIndex] = useState(0);

  const currentEstimand = estimandsData.find(e => e.id === estimandId) || estimandsData[0];
  
  // Calculate total slides dynamically based on content
  const assumptionSlides = Math.max(1, Math.ceil(currentEstimand.assumptions.length / 4));
  const estimatorSlides = Math.ceil(currentEstimand.estimators.length / 6);
  const referenceSlides = Math.ceil(currentEstimand.references.length / 3);
  const totalSlides = 1 + 1 + assumptionSlides + estimatorSlides + 2 + referenceSlides; // title + definition + assumptions + estimators + 2 code + references

  // Reset slide index when estimand changes
  useEffect(() => {
    setSlideIndex(0);
  }, [estimandId]);

  const goToNext = () => {
    if (slideIndex < totalSlides - 1) {
      setSlideIndex(slideIndex + 1);
    } else {
      // Move to next estimand
      const currentIdx = estimandsData.findIndex(e => e.id === estimandId);
      if (currentIdx < estimandsData.length - 1) {
        setSearchParams({ id: estimandsData[currentIdx + 1].id });
        setSlideIndex(0);
      }
    }
  };

  const goToPrevious = () => {
    if (slideIndex > 0) {
      setSlideIndex(slideIndex - 1);
    } else {
      // Move to previous estimand
      const currentIdx = estimandsData.findIndex(e => e.id === estimandId);
      if (currentIdx > 0) {
        setSearchParams({ id: estimandsData[currentIdx - 1].id });
        setSlideIndex(totalSlides - 1);
      }
    }
  };

  const handleEstimandChange = (newEstimandId: string) => {
    setSearchParams({ id: newEstimandId });
    setSlideIndex(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="container mx-auto px-4 py-8">
        {/* Header with estimand selector */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <Presentation className="h-6 w-6 text-primary" />
            <Select value={estimandId} onValueChange={handleEstimandChange}>
              <SelectTrigger className="w-[500px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[400px]">
                {estimandsData.map(e => (
                  <SelectItem key={e.id} value={e.id}>
                    {e.short_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="text-sm text-muted-foreground">
            Slide {slideIndex + 1} of {totalSlides}
          </div>
        </div>

        {/* Slide (16:9 PowerPoint ratio) */}
        <div className="flex items-center justify-center mb-8">
          <EstimandSlideStandalone estimand={currentEstimand} slideIndex={slideIndex} />
        </div>

        {/* Navigation controls */}
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={goToPrevious}
            disabled={slideIndex === 0 && estimandsData.findIndex(e => e.id === estimandId) === 0}
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
            disabled={slideIndex === totalSlides - 1 && estimandsData.findIndex(e => e.id === estimandId) === estimandsData.length - 1}
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

export default SlidesView;
