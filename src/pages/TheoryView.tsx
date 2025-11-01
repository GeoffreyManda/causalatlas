import { useState, useEffect } from 'react';
import { useSearchParams, useLocation, useNavigate } from 'react-router-dom';
import Navigation from '@/components/Navigation';
import TheorySlide from '@/components/TheorySlide';
import { causalTheory } from '@/data/theory';
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, BookOpen, Download, Home, Network, GraduationCap, Target } from 'lucide-react';
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
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { estimandFamilies } from '@/data/estimandFamilies';
import { estimandsData } from '@/data/estimands';

// Combine original theory topics with new ones
const allTopics = [...allTheoryTopics, ...causalTheory];

const TheoryView = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const topicId = searchParams.get('id') || allTopics[0].id;
  const [slideIndex, setSlideIndex] = useState(0);
  const [selectedTier, setSelectedTier] = useState<string>('all');

  // Get referrer from state or default to learning hub
  const referrer = (location.state as any)?.from || '/learning';

  const currentTopic = allTopics.find(t => t.id === topicId) || allTopics[0];
  
  // Calculate total slides dynamically based on content
  const contentParagraphs = currentTopic.content.split('\n\n').filter(p => p.trim()).length;
  const objectiveSlides = Math.ceil(currentTopic.learningObjectives.length / 3);
  const definitionSlides = Math.ceil(currentTopic.keyDefinitions.length / 3);
  const contentSlides = Math.ceil(contentParagraphs / 3);
  const referenceSlides = Math.ceil(currentTopic.references.length / 3);
  const totalContentSlides = 1 + objectiveSlides + definitionSlides + contentSlides + 2 + referenceSlides; // title + objectives + definitions + content + 2 code + references
  const totalSlides = totalContentSlides + 1; // +1 for navigation slide

  // Filter topics
  const filteredTopics = allTopics.filter(t => {
    if (selectedTier !== 'all' && t.tier !== selectedTier) return false;
    return true;
  });

  const tiers = ['all', 'Foundational', 'Intermediate', 'Advanced', 'Frontier'];

  // Group topics and estimands by tier for nested menus
  const theoryByTier = {
    Foundational: allTopics.filter(t => t.tier === 'Foundational'),
    Intermediate: allTopics.filter(t => t.tier === 'Intermediate'),
    Advanced: allTopics.filter(t => t.tier === 'Advanced'),
    Frontier: allTopics.filter(t => t.tier === 'Frontier'),
  };

  const estimandsByTier = {
    Basic: estimandsData.filter(e => e.tier === 'Basic'),
    Intermediate: estimandsData.filter(e => e.tier === 'Intermediate'),
    Advanced: estimandsData.filter(e => e.tier === 'Advanced'),
    Frontier: estimandsData.filter(e => e.tier === 'Frontier'),
  };

  // Reset slide index when topic changes or restart param
  useEffect(() => {
    if (searchParams.get('restart') === 'true') {
      setSlideIndex(0);
      setSearchParams({ id: topicId }); // Remove restart param
    } else {
      setSlideIndex(0);
    }
  }, [topicId]);

  const goToNext = () => {
    if (slideIndex < totalSlides - 1) {
      setSlideIndex(slideIndex + 1);
    }
    // Removed auto-jump to next topic
  };

  const goToPrevious = () => {
    if (slideIndex > 0) {
      setSlideIndex(slideIndex - 1);
    }
    // Removed auto-jump to previous topic
  };

  const handleTopicChange = (newTopicId: string) => {
    setSearchParams({ id: newTopicId });
    setSlideIndex(0);
  };

  const downloadSlides = async () => {
    const loadingToast = toast.loading('Generating PDF...');
    
    try {
      // Only include content slides in PDF, exclude navigation slide
      await generateSlidesFromRenderer(
        (index) => setSlideIndex(index),
        totalContentSlides,
        'theory-slide-container',
        `${currentTopic.title.replace(/[^a-z0-9]/gi, '_')}_theory.pdf`,
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

  // Navigate to previous/next topic deck
  const currentIndex = allTopics.findIndex(t => t.id === topicId);
  const hasPreviousDeck = currentIndex > 0;
  const hasNextDeck = currentIndex < allTopics.length - 1;

  const goToPreviousDeck = () => {
    if (hasPreviousDeck) {
      const previousTopic = allTopics[currentIndex - 1];
      handleTopicChange(previousTopic.id);
    }
  };

  const goToNextDeck = () => {
    if (hasNextDeck) {
      const nextTopic = allTopics[currentIndex + 1];
      handleTopicChange(nextTopic.id);
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
                <BookOpen className="h-4 w-4" />
                Navigate To
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-64 bg-popover z-[100]">
              <DropdownMenuLabel>Main Sections</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => navigate('/')} className="gap-2 cursor-pointer">
                <Home className="h-4 w-4" />
                Home
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => navigate(`/network?node=${topicId}`)} className="gap-2 cursor-pointer">
                <Network className="h-4 w-4" />
                Network View (Return to Node)
              </DropdownMenuItem>
              
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Slide Navigation</DropdownMenuLabel>
              <DropdownMenuItem 
                onClick={goToPrevious} 
                disabled={slideIndex === 0}
                className="gap-2 cursor-pointer"
              >
                <ChevronLeft className="h-4 w-4" />
                Previous Slide
              </DropdownMenuItem>
              <DropdownMenuItem 
                onClick={goToNext} 
                disabled={slideIndex === totalSlides - 1}
                className="gap-2 cursor-pointer"
              >
                <ChevronRight className="h-4 w-4" />
                Next Slide
              </DropdownMenuItem>
              
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Related Content</DropdownMenuLabel>
              
              {/* Learning Hub with nested tiers */}
              <DropdownMenuSub>
                <DropdownMenuSubTrigger className="cursor-pointer">
                  <GraduationCap className="h-4 w-4 mr-2" />
                  Learning Hub
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-64 max-h-80 overflow-y-auto bg-popover z-[100]">
                  <DropdownMenuItem onClick={() => navigate('/learning')} className="cursor-pointer font-semibold">
                    View All Topics
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  {Object.entries(theoryByTier).map(([tier, topics]) => (
                    <DropdownMenuSub key={tier}>
                      <DropdownMenuSubTrigger className="cursor-pointer">
                        <Badge variant="outline" className="mr-2">{tier}</Badge>
                        {topics.length} topics
                      </DropdownMenuSubTrigger>
                      <DropdownMenuSubContent className="w-64 max-h-80 overflow-y-auto bg-popover z-[100]">
                        {topics.map(t => (
                          <DropdownMenuItem 
                            key={t.id} 
                            onClick={() => navigate(`/theory?id=${t.id}`)}
                            className="cursor-pointer"
                          >
                            {t.title}
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuSubContent>
                    </DropdownMenuSub>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>

              {/* Estimands Library with nested tiers */}
              <DropdownMenuSub>
                <DropdownMenuSubTrigger className="cursor-pointer">
                  <Target className="h-4 w-4 mr-2" />
                  Estimands Library
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-64 max-h-80 overflow-y-auto bg-popover z-[100]">
                  <DropdownMenuItem onClick={() => navigate('/estimands')} className="cursor-pointer font-semibold">
                    View All Estimands
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  {Object.entries(estimandsByTier).map(([tier, ests]) => (
                    <DropdownMenuSub key={tier}>
                      <DropdownMenuSubTrigger className="cursor-pointer">
                        <Badge variant="outline" className="mr-2">{tier}</Badge>
                        {ests.length} estimands
                      </DropdownMenuSubTrigger>
                      <DropdownMenuSubContent className="w-64 max-h-80 overflow-y-auto bg-popover z-[100]">
                        {ests.map(e => (
                          <DropdownMenuItem 
                            key={e.id} 
                            onClick={() => navigate(`/slides?estimand=${e.id}`)}
                            className="cursor-pointer"
                          >
                            {e.short_name}
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuSubContent>
                    </DropdownMenuSub>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>

              {/* Generated Slides with families */}
              <DropdownMenuSub>
                <DropdownMenuSubTrigger className="cursor-pointer">
                  <BookOpen className="h-4 w-4 mr-2" />
                  Generated Slides
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-72 max-h-80 overflow-y-auto bg-popover z-[100]">
                  <DropdownMenuItem onClick={() => navigate('/slides')} className="cursor-pointer font-semibold">
                    View All Families
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  {estimandFamilies.map(family => (
                    <DropdownMenuItem 
                      key={family.id} 
                      onClick={() => navigate(`/slides?family=${family.id}`)}
                      className="cursor-pointer"
                    >
                      <div className="flex flex-col">
                        <span className="font-medium">{family.title}</span>
                        <span className="text-xs text-muted-foreground">{family.description}</span>
                      </div>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
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
            Showing {filteredTopics.length} of {allTopics.length} topics
          </p>
        </div>

        {/* Header with topic selector */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <BookOpen className="h-6 w-6 text-primary" />
            <Select value={topicId} onValueChange={handleTopicChange}>
              <SelectTrigger className="w-full max-w-[400px] bg-card">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[400px] z-[100]">
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
        <div className="flex items-center justify-center mb-8" id="theory-slide-container">
          <TheorySlide 
            topic={currentTopic} 
            slideIndex={slideIndex}
            totalContentSlides={totalContentSlides}
            totalSlides={totalSlides}
            onNavigate={navigate}
            onSlideChange={setSlideIndex}
            topicId={topicId}
            onDownload={downloadSlides}
            onPreviousDeck={goToPreviousDeck}
            onNextDeck={goToNextDeck}
            hasPreviousDeck={hasPreviousDeck}
            hasNextDeck={hasNextDeck}
          />
        </div>

        {/* Navigation */}
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
      </div>
    </div>
  );
};

export default TheoryView;
