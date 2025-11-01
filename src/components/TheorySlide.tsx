import { TheoryTopic } from '@/data/theory';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BookOpen, Target, Lightbulb, Home, Network, GraduationCap, ChevronLeft } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import 'katex/dist/katex.min.css';

interface TheorySlideProps {
  topic: TheoryTopic;
  slideIndex: number;
  totalContentSlides?: number;
  onNavigate?: (path: string) => void;
  topicId?: string;
}

const TheorySlide = ({ topic, slideIndex, totalContentSlides, onNavigate, topicId }: TheorySlideProps) => {
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
  if (totalContentSlides !== undefined && slideIndex === totalContentSlides && onNavigate) {
    return (
      <div className="w-full aspect-[16/9] bg-gradient-to-br from-primary/10 via-background to-primary/5 rounded-xl shadow-2xl p-12">
        <div className="h-full flex flex-col items-center justify-center">
          <h2 className="text-5xl font-bold mb-4 text-center">End of Slides</h2>
          <p className="text-center text-muted-foreground mb-8 text-xl">
            You've reached the last slide. Where would you like to go next?
          </p>
          <div className="grid grid-cols-3 gap-4 w-full max-w-3xl">
            <Button onClick={() => onNavigate('/')} variant="outline" className="gap-2 h-16 text-lg">
              <Home className="h-5 w-5" />
              Home
            </Button>
            <Button onClick={() => onNavigate('/learning')} variant="outline" className="gap-2 h-16 text-lg">
              <GraduationCap className="h-5 w-5" />
              Learning Hub
            </Button>
            <Button onClick={() => onNavigate(`/network?node=${topicId}`)} variant="default" className="gap-2 h-16 text-lg">
              <Network className="h-5 w-5" />
              Back to Network
            </Button>
            <Button onClick={() => onNavigate('/estimands')} variant="outline" className="gap-2 h-16 text-lg">
              <Target className="h-5 w-5" />
              Estimands Library
            </Button>
            <Button onClick={() => onNavigate('/slides')} variant="outline" className="gap-2 h-16 text-lg">
              <BookOpen className="h-5 w-5" />
              Generated Slides
            </Button>
            <Button onClick={() => onNavigate(`/theory?id=${topicId}&restart=true`)} variant="outline" className="gap-2 h-16 text-lg">
              <ChevronLeft className="h-5 w-5" />
              Restart Slides
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default TheorySlide;
