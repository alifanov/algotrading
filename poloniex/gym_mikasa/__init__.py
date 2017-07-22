from gym.envs.registration import register

register(
    id='mikasa-v0',
    entry_point='gym_mikasa.envs:MikasaEnv',
)