set pagination off
set confirm off
set print pretty on
set breakpoint pending off
set $dogfight_break_tick = -1
set $dogfight_watch_reward = 0
set $dogfight_watch_reward_set = 0

break dogfight_debug_anchor_reset
commands
silent
printf "\n===== dogfight_debug_anchor_reset =====\n"
bt 8
info args
info locals
printf "tick=%d stage=%d reward=%g terminal=%g death=%d\n", env->tick, env->stage, env->rewards[0], env->terminals[0], env->death_reason
printf "player pos=(%g,%g,%g) vel=(%g,%g,%g) g=%g\n", env->player.pos.x, env->player.pos.y, env->player.pos.z, env->player.vel.x, env->player.vel.y, env->player.vel.z, env->player.g_force
printf "opponent pos=(%g,%g,%g) vel=(%g,%g,%g)\n", env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z, env->opponent.vel.x, env->opponent.vel.y, env->opponent.vel.z
continue
end

break dogfight_debug_anchor_pre_step
commands
silent
if $dogfight_break_tick >= 0 && step_index == $dogfight_break_tick
  printf "\n===== requested break tick %d =====\n", step_index
  bt 16
  info args
  info locals
  printf "tick=%d stage=%d reward=%g terminal=%g death=%d\n", env->tick, env->stage, env->rewards[0], env->terminals[0], env->death_reason
  printf "action=(%g,%g,%g,%g,%g)\n", action[0], action[1], action[2], action[3], action[4]
  printf "player pos=(%g,%g,%g) vel=(%g,%g,%g) g=%g\n", env->player.pos.x, env->player.pos.y, env->player.pos.z, env->player.vel.x, env->player.vel.y, env->player.vel.z, env->player.g_force
  printf "opponent pos=(%g,%g,%g) vel=(%g,%g,%g)\n", env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z, env->opponent.vel.x, env->opponent.vel.y, env->opponent.vel.z
else
  printf "\n===== dogfight_debug_anchor_pre_step step=%d =====\n", step_index
  bt 8
  info args
  info locals
  printf "tick=%d stage=%d action=(%g,%g,%g,%g,%g)\n", env->tick, env->stage, action[0], action[1], action[2], action[3], action[4]
  continue
end
end

break dogfight_debug_anchor_post_step
commands
silent
printf "\n===== dogfight_debug_anchor_post_step step=%d =====\n", step_index
bt 8
info args
info locals
printf "tick=%d stage=%d reward=%g terminal=%g death=%d episode_return=%g\n", env->tick, env->stage, reward, terminal, env->death_reason, env->episode_return
printf "player pos=(%g,%g,%g) vel=(%g,%g,%g) g=%g\n", env->player.pos.x, env->player.pos.y, env->player.pos.z, env->player.vel.x, env->player.vel.y, env->player.vel.z, env->player.g_force
printf "opponent pos=(%g,%g,%g) vel=(%g,%g,%g)\n", env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z, env->opponent.vel.x, env->opponent.vel.y, env->opponent.vel.z
continue
end

break dogfight_debug_anchor_terminal
commands
silent
printf "\n===== dogfight_debug_anchor_terminal step=%d reason=%d =====\n", step_index, reason
bt 12
info args
info locals
printf "tick=%d stage=%d reward=%g terminal=%g death=%d kill=%d opp_kill=%d\n", env->tick, env->stage, env->rewards[0], env->terminals[0], env->death_reason, env->kill, env->opp_kill
continue
end

break c_reset
commands
silent
printf "\n===== c_reset =====\n"
bt 8
info args
info locals
continue
end

break c_step
commands
silent
printf "\n===== c_step entry =====\n"
bt 8
info args
info locals
printf "tick=%d stage=%d action=(%g,%g,%g,%g,%g)\n", env->tick, env->stage, env->actions[0], env->actions[1], env->actions[2], env->actions[3], env->actions[4]
if $dogfight_watch_reward && !$dogfight_watch_reward_set
  watch -location env->rewards[0]
  set $dogfight_watch_reward_set = 1
  printf "reward watchpoint installed at tick=%d\n", env->tick
end
continue
end

break spawn_by_curriculum
commands
silent
printf "\n===== spawn_by_curriculum =====\n"
bt 8
info args
info locals
printf "tick=%d stage=%d curriculum_target=%g\n", env->tick, env->stage, env->curriculum_target
continue
end

break compute_observations
commands
silent
printf "\n===== compute_observations =====\n"
bt 6
info args
info locals
printf "tick=%d stage=%d obs_scheme=%d obs_size=%d\n", env->tick, env->stage, env->obs_scheme, env->obs_size
continue
end

break check_hit
commands
silent
printf "\n===== check_hit =====\n"
bt 8
info args
info locals
continue
end
