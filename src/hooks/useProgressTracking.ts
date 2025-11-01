import { useEffect, useState } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

export const useProgressTracking = () => {
  const { toast } = useToast();
  const [userId, setUserId] = useState<string | null>(null);

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setUserId(user?.id || null);
    });
  }, []);

  const trackProgress = async (contentType: 'theory' | 'estimand' | 'slide', contentId: string, xpEarned: number = 50) => {
    if (!userId) return;

    try {
      // Insert progress
      const { error: progressError } = await supabase
        .from('user_progress')
        .insert({
          user_id: userId,
          content_type: contentType,
          content_id: contentId,
          xp_earned: xpEarned
        });

      if (progressError && !progressError.message.includes('duplicate')) {
        throw progressError;
      }

      // Update total XP
      const { data: profile } = await supabase
        .from('profiles')
        .select('total_xp, current_level')
        .eq('id', userId)
        .single();

      if (profile) {
        const newTotalXp = profile.total_xp + xpEarned;
        let newLevel = profile.current_level;

        // Check level ups
        if (newTotalXp >= 50000) newLevel = 'master';
        else if (newTotalXp >= 15000) newLevel = 'expert';
        else if (newTotalXp >= 5000) newLevel = 'advanced';
        else if (newTotalXp >= 1000) newLevel = 'intermediate';

        await supabase
          .from('profiles')
          .update({ 
            total_xp: newTotalXp,
            current_level: newLevel
          })
          .eq('id', userId);

        if (newLevel !== profile.current_level) {
          toast({
            title: '🎉 Level Up!',
            description: `You've reached ${newLevel} level!`,
            duration: 5000
          });
        }

        // Check achievements
        await checkAchievements(userId, contentType, newTotalXp);

        toast({
          title: `+${xpEarned} XP`,
          description: 'Progress saved!',
          duration: 2000
        });
      }
    } catch (error) {
      console.error('Error tracking progress:', error);
    }
  };

  const checkAchievements = async (userId: string, contentType: string, totalXp: number) => {
    // Get user's completed content count
    const { data: progressData } = await supabase
      .from('user_progress')
      .select('id')
      .eq('user_id', userId)
      .eq('content_type', contentType);

    const completedCount = progressData?.length || 0;

    // Get existing achievements
    const { data: existingAchievements } = await supabase
      .from('user_achievements')
      .select('achievement_id')
      .eq('user_id', userId);

    const unlockedIds = new Set(existingAchievements?.map(a => a.achievement_id) || []);

    // Get all achievements
    const { data: achievements } = await supabase
      .from('achievements')
      .select('*');

    // Check which achievements should be unlocked
    for (const achievement of achievements || []) {
      if (unlockedIds.has(achievement.id)) continue;

      let shouldUnlock = false;

      if (achievement.requirement_type === 'total_xp' && totalXp >= achievement.requirement_value) {
        shouldUnlock = true;
      } else if (achievement.requirement_type === 'theory_complete' && contentType === 'theory' && completedCount >= achievement.requirement_value) {
        shouldUnlock = true;
      } else if (achievement.requirement_type === 'estimand_complete' && contentType === 'estimand' && completedCount >= achievement.requirement_value) {
        shouldUnlock = true;
      }

      if (shouldUnlock) {
        const { error } = await supabase
          .from('user_achievements')
          .insert({
            user_id: userId,
            achievement_id: achievement.id
          });

        if (!error) {
          toast({
            title: `🏆 Achievement Unlocked!`,
            description: `${achievement.icon} ${achievement.title}`,
            duration: 5000
          });

          // Add achievement XP
          await supabase
            .from('profiles')
            .update({ 
              total_xp: totalXp + achievement.xp_reward
            })
            .eq('id', userId);
        }
      }
    }
  };

  return { trackProgress, userId };
};
