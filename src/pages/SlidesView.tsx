import { useState } from 'react';
import { estimandsData } from '@/data/estimands';
import EstimandSlide from '@/components/EstimandSlide';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const SlidesView = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isExpertMode, setIsExpertMode] = useState(false);

  const currentEstimand = estimandsData[currentIndex];

  const goToNext = () => {
    if (currentIndex < estimandsData.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goToPrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="container mx-auto px-4 py-8">
        {/* Header with mode toggle */}
        <div className="flex items-center justify-between mb-6">
          <div className="text-sm text-muted-foreground">
            Slide {currentIndex + 1} of {estimandsData.length}
          </div>
          <Button
            variant={isExpertMode ? "default" : "outline"}
            onClick={() => setIsExpertMode(!isExpertMode)}
            className="font-semibold"
          >
            {isExpertMode ? '🎓 Expert Mode' : '📘 Basic Mode'}
          </Button>
        </div>

        {/* Main slide content */}
        <EstimandSlide 
          estimand={currentEstimand} 
          isExpertMode={isExpertMode}
        />

        {/* Navigation controls */}
        <div className="flex items-center justify-between mt-8">
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
            {estimandsData.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentIndex(idx)}
                className={`w-2 h-2 rounded-full transition-all ${
                  idx === currentIndex 
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
            disabled={currentIndex === estimandsData.length - 1}
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
