import { TheoryTopic } from '@/data/theory';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BookOpen, Target, Lightbulb, Home, Network, GraduationCap, ChevronLeft, ChevronRight, Download } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import 'katex/dist/katex.min.css';
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
import { allTheoryTopics } from '@/data/allTheoryTopics';
import { estimandFamilies } from '@/data/estimandFamilies';
import { estimandsData } from '@/data/estimands';

interface TheorySlideProps {
  topic: TheoryTopic;
  slideIndex: number;
  totalContentSlides?: number;
  totalSlides?: number;
  onNavigate?: (path: string) => void;
  onSlideChange?: (index: number) => void;
  topicId?: string;
  onDownload?: () => void;
  onPreviousDeck?: () => void;
  onNextDeck?: () => void;
  hasPreviousDeck?: boolean;
  hasNextDeck?: boolean;
}

const TheorySlide = ({ topic, slideIndex, totalContentSlides, totalSlides, onNavigate, onSlideChange, topicId, onDownload, onPreviousDeck, onNextDeck, hasPreviousDeck, hasNextDeck }: TheorySlideProps) => {
  // Group topics and estimands by tier
  const theoryByTier = {
    Foundational: allTheoryTopics.filter(t => t.tier === 'Foundational'),
    Intermediate: allTheoryTopics.filter(t => t.tier === 'Intermediate'),
    Advanced: allTheoryTopics.filter(t => t.tier === 'Advanced'),
  };

  const estimandsByTier = {
    Basic: estimandsData.filter(e => e.tier === 'Basic'),
    Intermediate: estimandsData.filter(e => e.tier === 'Intermediate'),
    Advanced: estimandsData.filter(e => e.tier === 'Advanced'),
    Frontier: estimandsData.filter(e => e.tier === 'Frontier'),
  };
  // Helper: Split content into chunks
  const chunkArray = <T,>(arr: T[], size: number): T[][] => {
    const chunks: T[][] = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  };

  // Split content into manageable sections
  const contentParagraphs = topic.content.split('\n\n').filter(p => p.trim());
  const contentChunks = chunkArray(contentParagraphs, 3); // 3 paragraphs per slide
  const objectiveChunks = chunkArray(topic.learningObjectives, 3); // Max 3 objectives per slide
  const definitionChunks = chunkArray(topic.keyDefinitions, 3); // Max 3 definitions per slide
  const referenceChunks = chunkArray(topic.references, 3); // Max 3 references per slide

  // Calculate slide positions
  let currentSlide = 0;
  
  // Slide 0: Title
  if (slideIndex === currentSlide++) {
    return (
      <div className="w-full aspect-[16/9] bg-gradient-to-br from-primary via-primary/90 to-primary/70 rounded-xl shadow-2xl p-12 flex flex-col items-center justify-center text-center">
        <div className="max-w-4xl">
          <Badge className="mb-6 text-lg px-6 py-2 bg-white text-primary">
            {topic.tier} Theory
          </Badge>
          <h1 className="text-6xl font-bold text-white mb-6 leading-tight">
            {topic.title}
          </h1>
          <p className="text-2xl text-white/90 mb-8">
            {topic.description}
          </p>
          {topic.prerequisites.length > 0 && (
            <div className="text-white/80 text-lg">
              Prerequisites: {topic.prerequisites.join(', ')}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Slides for Learning Objectives (1 slide per chunk)
  for (let i = 0; i < objectiveChunks.length; i++) {
    if (slideIndex === currentSlide++) {
      return (
        <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
          <div className="flex items-center gap-3 mb-8">
            <Target className="h-10 w-10 text-primary" />
            <h2 className="text-4xl font-bold">
              Learning Objectives
              {objectiveChunks.length > 1 && ` (${i + 1}/${objectiveChunks.length})`}
            </h2>
          </div>
          <div className="grid gap-6">
            {objectiveChunks[i].map((obj, idx) => (
              <Card key={idx} className="p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <span className="text-primary font-bold text-lg">{i * 3 + idx + 1}</span>
                  </div>
                  <div className="text-2xl text-foreground/90 leading-relaxed prose prose-lg max-w-none">
                    <ReactMarkdown>{obj}</ReactMarkdown>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      );
    }
  }

  // Slides for Key Definitions (1 slide per chunk)
  for (let i = 0; i < definitionChunks.length; i++) {
    if (slideIndex === currentSlide++) {
      return (
        <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
          <div className="flex items-center gap-3 mb-8">
            <BookOpen className="h-10 w-10 text-primary" />
            <h2 className="text-4xl font-bold">
              Key Definitions
              {definitionChunks.length > 1 && ` (${i + 1}/${definitionChunks.length})`}
            </h2>
          </div>
          <div className="grid gap-6">
            {definitionChunks[i].map((def, idx) => (
              <Card key={idx} className="p-6 bg-gradient-to-r from-card to-muted/20 border-l-4 border-primary">
                <h3 className="text-2xl font-bold text-primary mb-3">{def.term}</h3>
                <div className="text-xl text-muted-foreground leading-relaxed prose prose-lg max-w-none">
                  <ReactMarkdown>{def.definition}</ReactMarkdown>
                </div>
              </Card>
            ))}
          </div>
        </div>
      );
    }
  }

  // Slides for Main Content (1 slide per chunk)
  for (let i = 0; i < contentChunks.length; i++) {
    if (slideIndex === currentSlide++) {
      return (
        <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
          <div className="flex items-center gap-3 mb-6">
            <Lightbulb className="h-10 w-10 text-primary" />
            <h2 className="text-4xl font-bold">
              {topic.title}
              {contentChunks.length > 1 && ` (${i + 1}/${contentChunks.length})`}
            </h2>
          </div>
          <div className="prose prose-lg max-w-none text-lg leading-relaxed space-y-4 text-foreground/90">
            {contentChunks[i].map((para, idx) => (
              <ReactMarkdown key={idx}>{para}</ReactMarkdown>
            ))}
          </div>
        </div>
      );
    }
  }

  // Code Example Slide - Python
  if (slideIndex === currentSlide++) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <h2 className="text-4xl font-bold mb-6">Code Example: Python</h2>
        <div className="bg-terminal-bg rounded-lg p-6 h-[calc(100%-5rem)] overflow-auto">
          <pre className="text-terminal-fg font-mono text-sm leading-relaxed">
            <code>{topic.examples.python}</code>
          </pre>
        </div>
      </div>
    );
  }

  // Code Example Slide - R
  if (slideIndex === currentSlide++) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <h2 className="text-4xl font-bold mb-6">Code Example: R</h2>
        <div className="bg-terminal-bg rounded-lg p-6 h-[calc(100%-5rem)] overflow-auto">
          <pre className="text-terminal-fg font-mono text-sm leading-relaxed">
            <code>{topic.examples.r}</code>
          </pre>
        </div>
      </div>
    );
  }

  // Slides for References (1 slide per chunk)
  for (let i = 0; i < referenceChunks.length; i++) {
    if (slideIndex === currentSlide++) {
      return (
        <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
          <h2 className="text-4xl font-bold mb-8">
            References
            {referenceChunks.length > 1 && ` (${i + 1}/${referenceChunks.length})`}
          </h2>
          <div className="space-y-6">
            {referenceChunks[i].map((ref, idx) => (
              <Card key={idx} className="p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <span className="text-primary font-bold">[{i * 3 + idx + 1}]</span>
                  </div>
                  <div>
                    <p className="text-xl font-semibold mb-2">{ref.title}</p>
                    <p className="text-lg text-muted-foreground">
                      {ref.authors} ({ref.year})
                    </p>
                    <p className="text-sm text-primary mt-2">DOI: {ref.doi}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      );
    }
  }

  // Navigation Slide (final slide, non-downloadable)
  if (totalContentSlides !== undefined && slideIndex === totalContentSlides && onNavigate && onSlideChange && totalSlides) {
    return (
      <div className="w-full aspect-[16/9] bg-gradient-to-br from-primary/10 via-background to-primary/5 rounded-xl shadow-2xl p-8">
        <div className="h-full flex flex-col items-center justify-center">
          <h2 className="text-5xl font-bold mb-3 text-center">End of Slides</h2>
          <p className="text-center text-muted-foreground mb-6 text-lg">
            You've reached the last slide. Where would you like to go next?
          </p>
          
          {/* Download PDF Button */}
          {onDownload && (
            <div className="mb-4">
              <Button 
                onClick={onDownload}
                variant="default"
                size="lg"
                className="gap-2 h-16 px-8 text-lg"
              >
                <Download className="h-5 w-5" />
                Download Slides as PDF
              </Button>
            </div>
          )}

          <div className="grid grid-cols-4 gap-3 w-full max-w-5xl mb-6">
            {/* Slide Deck Navigation */}
            <Button 
              onClick={onPreviousDeck}
              disabled={!hasPreviousDeck}
              variant="outline" 
              className="gap-2 h-14"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous Slide Deck
            </Button>
            
            <Button 
              onClick={() => onSlideChange(0)} 
              variant="outline" 
              className="gap-2 h-14"
            >
              <ChevronLeft className="h-4 w-4" />
              Restart Slides
            </Button>

            <Button 
              onClick={onNextDeck}
              disabled={!hasNextDeck}
              variant="outline" 
              className="gap-2 h-14"
            >
              Next Slide Deck
              <ChevronRight className="h-4 w-4" />
            </Button>

            {/* Main Navigation */}
            <Button onClick={() => onNavigate('/')} variant="outline" className="gap-2 h-14">
              <Home className="h-4 w-4" />
              Home
            </Button>

            <Button onClick={() => onNavigate(`/network?node=${topicId}`)} variant="default" className="gap-2 h-14 col-span-4">
              <Network className="h-4 w-4" />
              Back to Network
            </Button>

            {/* Nested Dropdown Menus */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="gap-2 h-14 col-span-2">
                  <GraduationCap className="h-4 w-4" />
                  Learning Hub
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-64 max-h-96 overflow-y-auto bg-popover z-[100]">
                <DropdownMenuLabel>Theory Topics by Level</DropdownMenuLabel>
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
                          onClick={() => onNavigate(`/theory?id=${t.id}`)}
                          className="cursor-pointer"
                        >
                          {t.title}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="gap-2 h-14 col-span-2">
                  <Target className="h-4 w-4" />
                  Estimands Library
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-64 max-h-96 overflow-y-auto bg-popover z-[100]">
                <DropdownMenuLabel>Estimands by Level</DropdownMenuLabel>
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
                          onClick={() => onNavigate(`/slides?estimand=${e.id}`)}
                          className="cursor-pointer"
                        >
                          {e.short_name}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="gap-2 h-14 col-span-4">
                  <BookOpen className="h-4 w-4" />
                  Generated Slides
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-72 max-h-96 overflow-y-auto bg-popover z-[100]">
                <DropdownMenuLabel>Slides by Family</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {estimandFamilies.map(family => (
                  <DropdownMenuItem 
                    key={family.id} 
                    onClick={() => onNavigate(`/slides?family=${family.id}`)}
                    className="cursor-pointer"
                  >
                    <div className="flex flex-col">
                      <span className="font-medium">{family.title}</span>
                      <span className="text-xs text-muted-foreground">{family.description}</span>
                    </div>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default TheorySlide;
