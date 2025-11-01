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
} from "@/components/ui/dropdown-menu";

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
  const totalSlides = 1 + objectiveSlides + definitionSlides + contentSlides + 2 + referenceSlides; // title + objectives + definitions + content + 2 code + references

  // Filter topics
  const filteredTopics = allTopics.filter(t => {
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
      await generateSlidesFromRenderer(
        (index) => setSlideIndex(index),
        totalSlides,
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
              <DropdownMenuItem onClick={() => navigate(`/network?node=${topicId}`)} className="gap-2 cursor-pointer">
                <Network className="h-4 w-4" />
                Network View (Return to Node)
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Related Content</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => navigate('/estimands')} className="gap-2 cursor-pointer">
                <Target className="h-4 w-4" />
                Estimands Library
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => navigate('/slides')} className="gap-2 cursor-pointer">
                <BookOpen className="h-4 w-4" />
                Generated Slides
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
          <TheorySlide topic={currentTopic} slideIndex={slideIndex} />
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
              <Button onClick={() => navigate(`/network?node=${topicId}`)} variant="default" className="gap-2">
                <Network className="h-4 w-4" />
                Back to Network
              </Button>
              <Button onClick={() => navigate('/estimands')} variant="outline" className="gap-2">
                <Target className="h-4 w-4" />
                Estimands Library
              </Button>
              <Button onClick={() => navigate('/slides')} variant="outline" className="gap-2">
                <BookOpen className="h-4 w-4" />
                Generated Slides
              </Button>
              <Button onClick={() => setSlideIndex(0)} variant="outline" className="gap-2">
                <ChevronLeft className="h-4 w-4" />
                Restart Slides
              </Button>
            </div>
          </div>
        )}

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
