import { estimandsData } from './estimands';
import { allTheoryTopics } from './allTheoryTopics';
import { dagLessons } from './dagLessons';
import comprehensiveLessons from './comprehensiveLessons';

export interface Lesson {
  id: string;
  title: string;
  description: string;
  category: 'estimand' | 'theory';
  tier: string;
  pythonCode: string;
  rCode: string;
  expectedOutput?: string;
  learningObjectives?: string[];
}

// Convert estimands to lessons
export const estimandLessons: Lesson[] = estimandsData.map(estimand => ({
  id: `estimand-${estimand.id}`,
  title: estimand.short_name,
  description: `Framework: ${estimand.framework} | Design: ${estimand.design}`,
  category: 'estimand' as const,
  tier: estimand.tier,
  pythonCode: estimand.examples.python,
  rCode: estimand.examples.r,
  learningObjectives: estimand.assumptions.slice(0, 3)
}));

// Convert theory topics to lessons
export const theoryLessons: Lesson[] = allTheoryTopics
  .filter(topic => topic.examples?.python && topic.examples?.r)
  .map(topic => ({
    id: `theory-${topic.id}`,
    title: topic.title,
    description: topic.description,
    category: 'theory' as const,
    tier: topic.tier,
    pythonCode: topic.examples.python,
    rCode: topic.examples.r,
    learningObjectives: topic.learningObjectives
  }));

// All lessons combined
export const allLessons: Lesson[] = [
  ...comprehensiveLessons,  // Comprehensive lessons from "What If" and "Causality" books
  ...dagLessons,
  ...theoryLessons,
  ...estimandLessons
];

// Group lessons by tier
export const lessonsByTier = {
  Foundational: allLessons.filter(l => l.tier === 'Foundational' || l.tier === 'Basic'),
  Intermediate: allLessons.filter(l => l.tier === 'Intermediate'),
  Advanced: allLessons.filter(l => l.tier === 'Advanced'),
  Frontier: allLessons.filter(l => l.tier === 'Frontier')
};
