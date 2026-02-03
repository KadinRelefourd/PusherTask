from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import DoubleToFloat, StepCounter, RewardSum, Compose, InitTracker

def make_env(env_name, device, max_steps=200, render_mode=None):
    base = GymEnv(env_name, device=device, render_mode=render_mode)
    return TransformedEnv(
        base,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            StepCounter(max_steps=max_steps),
            RewardSum(),  # adds "episode_reward" when episode ends
        ),
    )
