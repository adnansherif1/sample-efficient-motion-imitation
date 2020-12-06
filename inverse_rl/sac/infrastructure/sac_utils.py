from gym.envs.registration import register
def register_custom_envs():
    from gym.envs.registration import registry
    if 'DanceRev-HH-v0' not in registry.env_specs:
        register(
            id='DanceRev-HH-v0',
            entry_point='sac.envs.dance_rev.env:HipHop',
            max_episode_steps=24*30,
            reward_threshold=200,
        )