import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Lock } from 'lucide-react';

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  xp_reward: number;
  tier: string;
}

interface AchievementBadgeProps {
  achievement: Achievement;
  unlocked?: boolean;
  unlockedAt?: string;
}

export const AchievementBadge = ({ achievement, unlocked, unlockedAt }: AchievementBadgeProps) => {
  return (
    <Card className={`p-4 transition-all ${unlocked ? 'bg-gradient-to-br from-primary/10 to-primary/5 border-primary/30' : 'opacity-50 grayscale'}`}>
      <div className="flex items-start gap-3">
        <div className="text-3xl">{unlocked ? achievement.icon : <Lock className="h-8 w-8 text-muted-foreground" />}</div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold">{achievement.title}</h3>
            <Badge variant="secondary" className="text-xs">{achievement.tier}</Badge>
          </div>
          <p className="text-sm text-muted-foreground mb-2">{achievement.description}</p>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-yellow-600">+{achievement.xp_reward} XP</span>
            {unlocked && unlockedAt && (
              <span className="text-xs text-muted-foreground">
                {new Date(unlockedAt).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};
