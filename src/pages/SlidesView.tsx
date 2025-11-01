import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate, useLocation } from 'react-router-dom';
import { estimandsData } from '@/data/estimands';
import EstimandSlideStandalone from '@/components/EstimandSlideStandalone';
import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Presentation, Download, ArrowLeft } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

const SlidesView = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const estimandId = searchParams.get('id') || estimandsData[0].id;
  const [slideIndex, setSlideIndex] = useState(0);
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');

  // Get referrer from state or default to estimands page
  const referrer = (location.state as any)?.from || '/estimands';

  const currentEstimand = estimandsData.find(e => e.id === estimandId) || estimandsData[0];
  
  // Calculate total slides dynamically based on content
  const assumptionSlides = Math.max(1, Math.ceil(currentEstimand.assumptions.length / 4));
  const estimatorSlides = Math.ceil(currentEstimand.estimators.length / 6);
  const referenceSlides = Math.ceil(currentEstimand.references.length / 3);
  const totalSlides = 1 + 1 + assumptionSlides + estimatorSlides + 2 + referenceSlides; // title + definition + assumptions + estimators + 2 code + references

  // Filter estimands
  const filteredEstimands = estimandsData.filter(e => {
    if (selectedTier !== 'all' && e.tier !== selectedTier) return false;
    if (selectedFramework !== 'all' && e.framework !== selectedFramework) return false;
    return true;
  });

  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(estimandsData.map(e => e.framework)))];

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

  const downloadSlides = () => {
    toast.info('Slide download feature coming soon (paywall feature)');
    // Future: Generate PDF of all slides for this estimand
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <Navigation />
      
      <div className="container mx-auto px-4 py-8">
        {/* Back button and header */}
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

        {/* Filters */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <div className="flex flex-col gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Filter by Tier</label>
              <div className="flex flex-wrap gap-2">
                {tiers.map(tier => (
                  <Badge
                    key={tier}
                    variant={selectedTier === tier ? 'default' : 'outline'}
                    className="cursor-pointer px-4 py-2 hover:scale-105 transition-transform"
                    onClick={() => setSelectedTier(tier)}
                  >
                    {tier === 'all' ? 'All Tiers' : tier}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Filter by Framework</label>
              <div className="flex flex-wrap gap-2">
                {frameworks.map(fw => (
                  <Badge
                    key={fw}
                    variant={selectedFramework === fw ? 'default' : 'outline'}
                    className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                    onClick={() => setSelectedFramework(fw)}
                  >
                    {fw === 'all' ? 'All Frameworks' : fw.replace(/([A-Z])/g, ' $1').trim()}
                  </Badge>
                ))}
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Showing {filteredEstimands.length} of {estimandsData.length} estimands
            </p>
          </div>
        </div>

        {/* Header with estimand selector */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <Presentation className="h-6 w-6 text-primary" />
            <Select value={estimandId} onValueChange={handleEstimandChange}>
              <SelectTrigger className="w-full max-w-[500px] bg-popover">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[400px] bg-popover z-50">
                {filteredEstimands.map(e => (
                  <SelectItem key={e.id} value={e.id}>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">{e.tier}</Badge>
                      {e.short_name}
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
