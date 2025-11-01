import { TheoryTopic } from '@/data/theory';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { BookOpen, Target, Lightbulb } from 'lucide-react';
import 'katex/dist/katex.min.css';

interface TheorySlideProps {
  topic: TheoryTopic;
  slideIndex: number; // 0 = title, 1 = overview, 2 = definitions, 3+ = content sections
}

const TheorySlide = ({ topic, slideIndex }: TheorySlideProps) => {
  // Title Slide
  if (slideIndex === 0) {
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

  // Learning Objectives Slide
  if (slideIndex === 1) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <div className="flex items-center gap-3 mb-8">
          <Target className="h-10 w-10 text-primary" />
          <h2 className="text-4xl font-bold">Learning Objectives</h2>
        </div>
        <div className="grid gap-6">
          {topic.learningObjectives.map((obj, idx) => (
            <Card key={idx} className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <span className="text-primary font-bold text-lg">{idx + 1}</span>
                </div>
                <p className="text-2xl text-foreground/90 leading-relaxed">{obj}</p>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // Key Definitions Slide
  if (slideIndex === 2) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <div className="flex items-center gap-3 mb-8">
          <BookOpen className="h-10 w-10 text-primary" />
          <h2 className="text-4xl font-bold">Key Definitions</h2>
        </div>
        <div className="grid gap-6">
          {topic.keyDefinitions.map((def, idx) => (
            <Card key={idx} className="p-6 bg-gradient-to-r from-card to-muted/20 border-l-4 border-primary">
              <h3 className="text-2xl font-bold text-primary mb-3">{def.term}</h3>
              <p className="text-xl text-muted-foreground leading-relaxed">{def.definition}</p>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // Main Content Slide
  if (slideIndex === 3) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <div className="flex items-center gap-3 mb-6">
          <Lightbulb className="h-10 w-10 text-primary" />
          <h2 className="text-4xl font-bold">{topic.title}</h2>
        </div>
        <div className="prose prose-lg max-w-none">
          <div className="text-lg leading-relaxed space-y-4 text-foreground/90">
            {topic.content.split('\n\n').slice(0, 4).map((para, idx) => (
              <p key={idx} className="whitespace-pre-wrap">{para}</p>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Code Example Slide
  if (slideIndex === 4) {
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

  // References Slide
  if (slideIndex === 5) {
    return (
      <div className="w-full aspect-[16/9] bg-background rounded-xl shadow-2xl p-12">
        <h2 className="text-4xl font-bold mb-8">References</h2>
        <div className="space-y-6">
          {topic.references.map((ref, idx) => (
            <Card key={idx} className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <span className="text-primary font-bold">[{idx + 1}]</span>
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

  return null;
};

export default TheorySlide;
