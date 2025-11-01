import { Estimand } from '@/data/estimands';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Activity, BookOpen, Code, FileText, Target } from 'lucide-react';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

interface EstimandSlideStandaloneProps {
  estimand: Estimand;
  slideIndex: number;
}

const EstimandSlideStandalone = ({ estimand, slideIndex }: EstimandSlideStandaloneProps) => {
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

  return null;
};

export default EstimandSlideStandalone;
