import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate, useLocation } from 'react-router-dom';
import { estimandsData } from '@/data/estimands';
import EstimandSlideStandalone from '@/components/EstimandSlideStandalone';
import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Presentation, Download, Home, Network, GraduationCap, Target, BookOpen } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { generateSlidesFromRenderer } from '@/lib/pdfGenerator';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const SlidesView = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const estimandId = searchParams.get('id') || estimandsData[0].id;
  const [slideIndex, setSlideIndex] = useState(0);
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [selectedDesign, setSelectedDesign] = useState<string>('all');
  const [selectedFamily, setSelectedFamily] = useState<string>('all');

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
    if (selectedDesign !== 'all' && e.design !== selectedDesign) return false;
    if (selectedFamily !== 'all' && e.estimand_family !== selectedFamily) return false;
    return true;
  });

  const tiers = ['all', 'Basic', 'Intermediate', 'Advanced', 'Frontier'];
  const frameworks = ['all', ...Array.from(new Set(estimandsData.map(e => e.framework)))];
  const designs = ['all', ...Array.from(new Set(estimandsData.map(e => e.design))).sort()];
  const families = ['all', ...Array.from(new Set(estimandsData.map(e => e.estimand_family))).sort()];

  // Reset slide index when estimand changes
  useEffect(() => {
    setSlideIndex(0);
  }, [estimandId]);

  const goToNext = () => {
    if (slideIndex < totalSlides - 1) {
      setSlideIndex(slideIndex + 1);
    }
    // Removed auto-jump to next estimand
  };

  const goToPrevious = () => {
    if (slideIndex > 0) {
      setSlideIndex(slideIndex - 1);
    }
    // Removed auto-jump to previous estimand
  };

  const handleEstimandChange = (newEstimandId: string) => {
    setSearchParams({ id: newEstimandId });
    setSlideIndex(0);
  };

  const downloadSlides = async () => {
    const loadingToast = toast.loading('Generating PDF...');
    
    try {
      await generateSlidesFromRenderer(
        (index) => setSlideIndex(index),
        totalSlides,
        'slide-container',
        `${currentEstimand.short_name.replace(/[^a-z0-9]/gi, '_')}_slides.pdf`,
        (current, total) => {
          toast.loading(`Generating PDF... ${current}/${total}`, { id: loadingToast });
        }
      );
      
      toast.success('PDF downloaded successfully!', { id: loadingToast });
    } catch (error) {
      console.error('PDF generation error:', error);
      toast.error('Failed to generate PDF', { id: loadingToast });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <Navigation />
      
      <div className="container mx-auto px-4 py-8">
        {/* Navigation menu and download */}
        <div className="flex items-center justify-between mb-6">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="gap-2">
                <Presentation className="h-4 w-4" />
                Navigate To
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuLabel>Main Sections</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => navigate('/')} className="gap-2 cursor-pointer">
                <Home className="h-4 w-4" />
                Home
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => navigate('/learning')} className="gap-2 cursor-pointer">
                <GraduationCap className="h-4 w-4" />
                Learning Hub
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => navigate(`/network?node=${estimandId}`)} className="gap-2 cursor-pointer">
                <Network className="h-4 w-4" />
                Network View (Return to Node)
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Related Content</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => navigate('/estimands')} className="gap-2 cursor-pointer">
                <Target className="h-4 w-4" />
                Estimands Library
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => navigate('/theory')} className="gap-2 cursor-pointer">
                <BookOpen className="h-4 w-4" />
                Theory Slides
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          
          <Button
            variant="outline"
            onClick={downloadSlides}
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            Download Slides as PDF
          </Button>
        </div>

        {/* Filters */}
        <div className="mb-6 p-4 rounded-lg border bg-card">
          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
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
                <label className="text-sm font-medium mb-2 block">Filter by Type</label>
                <div className="flex flex-wrap gap-2">
                  {families.map(family => (
                    <Badge
                      key={family}
                      variant={selectedFamily === family ? 'default' : 'outline'}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedFamily(family)}
                    >
                      {family === 'all' ? 'All Types' : family === 'SurvivalTimeToEvent' ? 'Survival/Time-to-Event' : family.replace(/([A-Z])/g, ' $1').trim()}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
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
              <div>
                <label className="text-sm font-medium mb-2 block">Filter by Study Design</label>
                <div className="flex flex-wrap gap-2">
                  {designs.map(design => (
                    <Badge
                      key={design}
                      variant={selectedDesign === design ? 'default' : 'outline'}
                      className="cursor-pointer px-3 py-1.5 text-xs hover:scale-105 transition-transform"
                      onClick={() => setSelectedDesign(design)}
                    >
                      {design === 'all' ? 'All Designs' : design.replace(/_/g, ' ')}
                    </Badge>
                  ))}
                </div>
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
              <SelectTrigger className="w-full max-w-[500px] bg-card">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[400px]">
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
        <div className="flex items-center justify-center mb-8" id="slide-container">
          <EstimandSlideStandalone estimand={currentEstimand} slideIndex={slideIndex} />
        </div>

        {/* End of slides navigation card */}
        {slideIndex === totalSlides - 1 && (
          <div className="mb-6 p-6 rounded-lg border-2 border-primary bg-card shadow-lg">
            <h3 className="text-xl font-bold mb-4 text-center">End of Slides</h3>
            <p className="text-center text-muted-foreground mb-6">
              You've reached the last slide. Where would you like to go next?
            </p>
            <div className="grid md:grid-cols-3 gap-3">
              <Button onClick={() => navigate('/')} variant="outline" className="gap-2">
                <Home className="h-4 w-4" />
                Home
              </Button>
              <Button onClick={() => navigate('/learning')} variant="outline" className="gap-2">
                <GraduationCap className="h-4 w-4" />
                Learning Hub
              </Button>
              <Button onClick={() => navigate(`/network?node=${estimandId}`)} variant="default" className="gap-2">
                <Network className="h-4 w-4" />
                Back to Network
              </Button>
              <Button onClick={() => navigate('/estimands')} variant="outline" className="gap-2">
                <Target className="h-4 w-4" />
                Estimands Library
              </Button>
              <Button onClick={() => navigate('/theory')} variant="outline" className="gap-2">
                <BookOpen className="h-4 w-4" />
                Theory Slides
              </Button>
              <Button onClick={() => setSlideIndex(0)} variant="outline" className="gap-2">
                <ChevronLeft className="h-4 w-4" />
                Restart Slides
              </Button>
            </div>
          </div>
        )}

        {/* Navigation controls */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Button
              variant="outline"
              onClick={goToPrevious}
              disabled={slideIndex === 0}
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
              disabled={slideIndex === totalSlides - 1}
              className="gap-2"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

          {/* Navigation Menu at Bottom */}
          <div className="flex justify-center">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="gap-2">
                  <Presentation className="h-4 w-4" />
                  Navigate To
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="center" className="w-56">
                <DropdownMenuLabel>Slide Navigation</DropdownMenuLabel>
                <DropdownMenuItem 
                  onClick={goToPrevious}
                  disabled={slideIndex === 0 && estimandsData.findIndex(e => e.id === estimandId) === 0}
                  className="gap-2 cursor-pointer"
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous Slide
                </DropdownMenuItem>
                <DropdownMenuItem 
                  onClick={goToNext}
                  disabled={slideIndex === totalSlides - 1 && estimandsData.findIndex(e => e.id === estimandId) === estimandsData.length - 1}
                  className="gap-2 cursor-pointer"
                >
                  <ChevronRight className="h-4 w-4" />
                  Next Slide
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Main Sections</DropdownMenuLabel>
                <DropdownMenuItem onClick={() => navigate('/')} className="gap-2 cursor-pointer">
                  <Home className="h-4 w-4" />
                  Home
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => navigate('/learning')} className="gap-2 cursor-pointer">
                  <GraduationCap className="h-4 w-4" />
                  Learning Hub
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => navigate('/network')} className="gap-2 cursor-pointer">
                  <Network className="h-4 w-4" />
                  Network View
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Related Content</DropdownMenuLabel>
                <DropdownMenuItem onClick={() => navigate('/estimands')} className="gap-2 cursor-pointer">
                  <Target className="h-4 w-4" />
                  Estimands Library
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => navigate('/theory')} className="gap-2 cursor-pointer">
                  <BookOpen className="h-4 w-4" />
                  Theory Slides
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SlidesView;
