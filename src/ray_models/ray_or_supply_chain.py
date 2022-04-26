import or_gym
from or_gym.utils import create_env
import ray
from ray.rllib import agents
from ray import tune
from tqdm import tqdm

def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name, 
        lambda env_name: env(env_name,
            env_config=env_config))

# Environment and RL Configuration Settings
env_name = 'InvManagement-v1'
env_config = {} # Change environment parameters here
rl_config = dict(
    env=env_name,
    num_workers=4,
    env_config=env_config,
    framework="torch",
    # simple_optimizer=True, # for single-agent
    num_gpus=1,
    # model=dict(
    #     vf_share_layers=False,
    #     fcnet_activation='relu',
    #     fcnet_hiddens=[256, 256]
    # ),
    lr=1e-5
)

# Register environment
register_env(env_name, env_config)

# Initialize Ray and Build Agent
ray.init(ignore_reinit_error=True)
agent = agents.ppo.PPOTrainer(env=env_name,
    config=rl_config)

results = []
for i in tqdm(range(500)):
    res = agent.train()
    results.append(res)
    if (i+1) % 5 == 0:
        print('\rIter: {}\t Reward: {:.2f}'.format(
                i+1, res['episode_reward_mean']), end='')
ray.shutdown()