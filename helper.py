from boardgame2 import ReversiEnv
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
import gym
import torch as th

class SelfPlayEnv(ReversiEnv):
    def __init__(self, board_shape=8, local_player=None, LocalPlayer=None, verbose=0):
        self.players = [-1, 1]
        self.verbose = verbose
        if local_player:
            self.local_player = local_player
        else:
            self.local_player = LocalPlayer(board_shape=board_shape, flatten_action=False)
        self.board_shape = board_shape
        super(SelfPlayEnv, self).__init__(board_shape=board_shape)
        
        self.action_space = gym.spaces.Discrete(board_shape**2)
        self.observation_space = gym.spaces.Box(-1, 1, (1,board_shape,board_shape))
         
        
    def play(self, observation):
        action = self.local_player.predict(observation)
        (observation, self.current_player_num), reward, done, _ = super(SelfPlayEnv, self).step(action)
        if self.verbose == 1:
            print(action, done)
        return (observation, self.current_player_num), reward, done
    
    def encode_observation(self, observation):
        return observation * self.current_player_num
    
    
    def reset(self):
        self.n_step = 0
        self.local_player_num = np.random.choice(self.players)
        self.local_player.player = self.local_player_num
        self.observation, self.current_player_num = super(SelfPlayEnv, self).reset()
        self.allow_pass = True
        if self.verbose:
            print(f'You play with {-1*self.local_player_num}')
            print('Initial board')
            print(self.observation)
            
        if self.current_player_num == self.local_player_num:   
            (self.observation, self.current_player_num), _, done = self.play(self.observation)
            assert done == False
            if self.verbose:
                print('Plays oponent:')
                print(self.observation)
        
        return self.encode_observation(self.observation)
    
    def encode_action(self, action):
        return [action // self.board_shape, action % self.board_shape]
    
    def decode_action(self, action):
        return action[0] * self.board_shape + action[1]

    def step(self, action):
        self.n_step += 1
        action = self.encode_action(action)
        
        (self.observation, self.current_player_num), reward, done, _ = super(SelfPlayEnv, self).step(action)   
        if self.verbose:
            print(f'Step: {self.n_step}')
            print(f'You play:')
            print(action)
            print(self.observation)
            
        while not done and (self.current_player_num == self.local_player_num):
            if self.verbose:
                print('Plays oponent:')
            (self.observation, self.current_player_num), reward, done = self.play(self.observation)
            if self.verbose:
                print(self.observation)
        
        encoded_observation = self.encode_observation(self.observation)
        total_pieces = encoded_observation[0].sum() + encoded_observation[1].sum()
        reward = -float(self.local_player_num*reward)

        if self.verbose:
            print(f'Reward: {reward}')
        return encoded_observation, reward, done, {} 

def make_reversi_vec_env(
    env_class,
    n_envs = 1,
    seed = None,
    start_index: int = 0,
    monitor_dir = None,
    wrapper_class = None,
    env_kwargs = None,
    vec_env_cls = None,
    vec_env_kwargs = None,
    monitor_kwargs = None,
    wrapper_kwargs = None,
):
   
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            env = env_class(**env_kwargs)
               
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)