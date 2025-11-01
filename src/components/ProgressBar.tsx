import { useEffect, useState } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { Progress } from '@/components/ui/progress';
import { Card } from '@/components/ui/card';
import { Trophy, Zap } from 'lucide-react';

interface UserProfile {
  current_level: string;
  total_xp: number;
}

export const ProgressBar = () => {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadProfile();

    const channel = supabase
      .channel('profile-changes')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'profiles' }, loadProfile)
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const loadProfile = async () => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      setLoading(false);
      return;
    }

    const { data } = await supabase
      .from('profiles')
      .select('current_level, total_xp')
      .eq('id', user.id)
      .single();

    setProfile(data);
    setLoading(false);
  };

  if (loading || !profile) return null;

  const levelThresholds = {
    beginner: 0,
    intermediate: 1000,
    advanced: 5000,
    expert: 15000,
    master: 50000
  };

  const currentLevel = profile.current_level as keyof typeof levelThresholds;
  const currentThreshold = levelThresholds[currentLevel];
  const nextLevel = Object.keys(levelThresholds).indexOf(currentLevel) + 1;
  const nextLevelName = Object.keys(levelThresholds)[nextLevel] || 'master';
  const nextThreshold = levelThresholds[nextLevelName as keyof typeof levelThresholds] || levelThresholds.master;

  const progress = ((profile.total_xp - currentThreshold) / (nextThreshold - currentThreshold)) * 100;

  return (
    <Card className="p-4 bg-gradient-to-r from-primary/10 to-primary/5 border-primary/20">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Trophy className="h-5 w-5 text-primary" />
          <span className="font-semibold capitalize">{currentLevel}</span>
        </div>
        <div className="flex items-center gap-1 text-sm">
          <Zap className="h-4 w-4 text-yellow-500" />
          <span className="font-bold">{profile.total_xp} XP</span>
        </div>
      </div>
      <Progress value={progress} className="h-2" />
      <p className="text-xs text-muted-foreground mt-2">
        {nextLevel < Object.keys(levelThresholds).length 
          ? `${nextThreshold - profile.total_xp} XP to ${nextLevelName}`
          : 'Max level reached!'}
      </p>
    </Card>
  );
};
