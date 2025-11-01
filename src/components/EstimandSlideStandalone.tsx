import { Estimand, estimandsData } from '@/data/estimands';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Activity, BookOpen, Code, FileText, Target, Home, Network, GraduationCap, ChevronLeft, ChevronRight, Download } from 'lucide-react';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
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

interface EstimandSlideStandaloneProps {
  estimand: Estimand;
  slideIndex: number;
  totalContentSlides?: number;
  totalSlides?: number;
  onNavigate?: (path: string) => void;
  onSlideChange?: (index: number) => void;
  estimandId?: string;
  onDownload?: () => void;
  onPreviousDeck?: () => void;
  onNextDeck?: () => void;
  hasPreviousDeck?: boolean;
  hasNextDeck?: boolean;
}

const EstimandSlideStandalone = ({ estimand, slideIndex, totalContentSlides, totalSlides, onNavigate, onSlideChange, estimandId, onDownload, onPreviousDeck, onNextDeck, hasPreviousDeck, hasNextDeck }: EstimandSlideStandaloneProps) => {
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
  // Helper: Split arrays into chunks
  const chunkArray = <T,>(arr: T[], size: number): T[][] => {
    const chunks: T[][] = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  };

  // Split content into manageable sections
  const assumptionChunks = chunkArray(estimand.assumptions, 4); // Max 4 assumptions per slide
  const estimatorChunks = chunkArray(estimand.estimators, 6); // Max 6 estimators per slide
  const referenceChunks = chunkArray(estimand.references, 3); // Max 3 references per slide

  // Calculate slide positions
  let currentSlide = 0;

  // Slide 0: Title
  if (slideIndex === currentSlide++) {
    return (
      <div className="w-full aspect-[16/9] bg-gradient-to-br from-primary via-primary/90 to-primary/70 rounded-xl shadow-2xl p-12 flex flex-col items-center justify-center text-center">
        <div className="max-w-4xl">
          <Badge className="mb-6 text-lg px-6 py-2 bg-white text-primary">
            {estimand.tier} | {estimand.framework}
          </Badge>
          <h1 className="text-6xl font-bold text-white mb-6 leading-tight">
            {estimand.short_name}
          </h1>
          <div className="flex items-center justify-center gap-4 text-white/90 text-xl mb-4">
            <Badge variant="outline" className="bg-white/10 text-white border-white/20 text-lg px-4 py-2">
              {estimand.design.replace(/_/g, ' ')}
            </Badge>
            <Badge variant="outline" className="bg-white/10 text-white border-white/20 text-lg px-4 py-2">
              {estimand.estimand_family}
            </Badge>
          </div>
          <p className="text-white/80 text-lg mt-6">
            {estimand.discovery_status === 'identifiable' ? '✓ Identifiable' : estimand.discovery_status === 'partially_identifiable' ? '⚠ Partially Identifiable' : '✗ Non-identifiable'}
            {' • '}
            EIF: {estimand.eif_status}
          </p>
        </div>
      </div>
    );
  }

  // Slide: Definition & Identification
  if (slideIndex === currentSlide++) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <div className="flex items-center gap-3 mb-8">
          <Target className="h-10 w-10 text-primary" />
          <h2 className="text-4xl font-bold">Estimand Definition</h2>
        </div>
        <div className="space-y-8">
          <Card className="p-8 bg-gradient-to-r from-primary/5 to-primary/10 border-l-4 border-primary">
            <h3 className="text-2xl font-bold mb-4 text-primary">Target Parameter</h3>
            <div className="text-3xl">
              <BlockMath math={estimand.definition_tex} />
            </div>
          </Card>
          <Card className="p-8 bg-muted/50">
            <h3 className="text-2xl font-bold mb-4">Identification Formula</h3>
            <div className="text-2xl">
              <BlockMath math={estimand.identification_formula_tex} />
            </div>
          </Card>
        </div>
      </div>
    );
  }

  // Slides for Assumptions (1 slide per chunk)
  for (let i = 0; i < assumptionChunks.length; i++) {
    if (slideIndex === currentSlide++) {
      return (
        <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
          <div className="flex items-center gap-3 mb-8">
            <FileText className="h-10 w-10 text-primary" />
            <h2 className="text-4xl font-bold">
              Identification Assumptions
              {assumptionChunks.length > 1 && ` (${i + 1}/${assumptionChunks.length})`}
            </h2>
          </div>
          <div className="grid gap-6">
            {assumptionChunks[i].map((assumption, idx) => (
              <Card key={idx} className="p-6 hover:shadow-lg transition-shadow bg-gradient-to-r from-card to-muted/20 border-l-4 border-primary/50">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <span className="text-primary font-bold text-xl">{i * 4 + idx + 1}</span>
                  </div>
                  <p className="text-2xl text-foreground/90 leading-relaxed pt-2">{assumption}</p>
                </div>
              </Card>
            ))}
          </div>
          {estimand.assumptions.length === 0 && i === 0 && (
            <p className="text-2xl text-muted-foreground text-center py-12">No assumptions required for this estimand</p>
          )}
        </div>
      );
    }
  }

  // Slides for Estimators (1 slide per chunk)
  for (let i = 0; i < estimatorChunks.length; i++) {
    if (slideIndex === currentSlide++) {
      return (
        <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
          <div className="flex items-center gap-3 mb-8">
            <Activity className="h-10 w-10 text-primary" />
            <h2 className="text-4xl font-bold">
              Statistical Estimators
              {estimatorChunks.length > 1 && ` (${i + 1}/${estimatorChunks.length})`}
            </h2>
          </div>
          <div className="grid grid-cols-2 gap-6">
            {estimatorChunks[i].map((est, idx) => (
              <Card key={idx} className="p-6 bg-gradient-to-br from-card via-card to-primary/5 hover:shadow-xl transition-all hover:scale-105">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <span className="text-primary font-bold text-2xl">{i * 6 + idx + 1}</span>
                  </div>
                  <h3 className="text-xl font-semibold">{est}</h3>
                </div>
              </Card>
            ))}
          </div>
          {i === estimatorChunks.length - 1 && (
            <div className="mt-8 p-6 bg-muted/30 rounded-lg">
              <p className="text-lg text-muted-foreground">
                <strong>Discovery Status:</strong> {estimand.discovery_status} • <strong>EIF:</strong> {estimand.eif_status}
              </p>
            </div>
          )}
        </div>
      );
    }
  }

  // Slide: Python Code
  if (slideIndex === currentSlide++) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <div className="flex items-center gap-3 mb-6">
          <Code className="h-10 w-10 text-primary" />
          <h2 className="text-4xl font-bold">Python Implementation</h2>
        </div>
        <div className="bg-terminal-bg rounded-lg p-6 h-[calc(100%-6rem)] overflow-auto">
          <pre className="text-terminal-fg font-mono text-sm leading-relaxed">
            <code>{estimand.examples.python}</code>
          </pre>
        </div>
      </div>
    );
  }

  // Slide: R Code
  if (slideIndex === currentSlide++) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <div className="flex items-center gap-3 mb-6">
          <Code className="h-10 w-10 text-primary" />
          <h2 className="text-4xl font-bold">R Implementation</h2>
        </div>
        <div className="bg-terminal-bg rounded-lg p-6 h-[calc(100%-6rem)] overflow-auto">
          <pre className="text-terminal-fg font-mono text-sm leading-relaxed">
            <code>{estimand.examples.r}</code>
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
          <div className="flex items-center gap-3 mb-8">
            <BookOpen className="h-10 w-10 text-primary" />
            <h2 className="text-4xl font-bold">
              Key References
              {referenceChunks.length > 1 && ` (${i + 1}/${referenceChunks.length})`}
            </h2>
          </div>
          <div className="space-y-6">
            {referenceChunks[i].map((ref, idx) => (
              <Card key={idx} className="p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <span className="text-primary font-bold text-xl">[{i * 3 + idx + 1}]</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-2xl font-semibold mb-2">{ref.title}</p>
                    <p className="text-xl text-muted-foreground mb-2">
                      {ref.authors} ({ref.year})
                    </p>
                    <a 
                      href={`https://doi.org/${ref.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-lg text-primary hover:underline"
                    >
                      DOI: {ref.doi}
                    </a>
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

            <Button onClick={() => onNavigate(`/network?node=${estimandId}`)} variant="default" className="gap-2 h-14 col-span-4">
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

export default EstimandSlideStandalone;
