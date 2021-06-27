import numpy as np
import torch as th
from boardgame2 import ReversiEnv


class TorchPlayer():
    def get_valid_actions(self, board):
        return self.env.get_valid((board, 1)).reshape(-1) 
    
    def __init__(self, model, player=1, board_shape=None, env=None, deterministic=True, flatten_action=True):
        self.player = player
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.board_shape = board_shape
        self.env = env
        self.model = model.policy
        self.model.set_get_valid_actions(self.get_valid_actions)
        self.model.eval()
        self.flatten_action = flatten_action
        
        self.deterministic = deterministic
    
    def predict(self, board):
        board = board * self.player
#         print(self.player)
#         print(board)
        model_out = self.model.predict(board.reshape(1, 1, *board.shape), deterministic=self.deterministic)
        action = model_out[0].item()
#         print(action)
        if self.flatten_action:
            return action
        else:
            return [action // self.board_shape, action % self.board_shape]
        
        
class GreedyPlayer():
    def __init__(self, player=1, board_shape=None, env=None, flatten_action=False):
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.env = env
        self.player = player
        self.flatten_action = flatten_action
        self.board_shape = self.env.board.shape[0]
    
    def predict(self, board):
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        moves_score = []
        for a in valid_actions:
            next_state, _, _, _ = self.env.next_step((board, self.player), a)
            moves_score.append(next_state[0].sum() * self.player)
        best_score = max(moves_score)
        best_actions = valid_actions[np.array(moves_score)==best_score]
#         action = valid_actions[np.argmax(moves_score)]
        action = best_actions[np.random.randint(len(best_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action
        

class RandomPlayer():
    def __init__(self, player=1, board_shape=None, env=None, flatten_action=False):
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.env = env
        self.player = player
        self.flatten_action = flatten_action
        self.board_shape = self.env.board.shape[0]
    
    def predict(self, board):
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        if len(valid_actions) == 0:
            action = self.env.PASS
        else:
            action = valid_actions[np.random.randint(len(valid_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action


class DictPolicyPlayer():
    def __init__(self, player=1, board_shape=4, env=None, flatten_action=False, dict_folder='mdp/pi_func_only_winner.npy'):
        self.pi_dict = np.load(dict_folder, allow_pickle=True).item()
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.player = player
        self.flatten_action = flatten_action
        self.board_shape = board_shape
    
    def predict(self, board):
        board_tuple = tuple((board * self.player).reshape(-1))
        action = self.pi_dict[board_tuple]
        if self.flatten_action:
            return action
        else:
            return [action // self.board_shape, action % self.board_shape]
        
from helper import SelfPlayEnv
def evaluate_player(player, local_player, N = 100, verbose=0, eps=1e-10):
    # Los resultados son porcentajes ganados por Player_1
    # Siempre juego con 1 por que el self play muestra el tablero como si fuera siempre el primer jugador
    wins_as_first = 0
    plays_as_first = 0
    wins_as_second = 0
    plays_as_second = 0
    ties_as_first = 0
    ties_as_second = 0
    self_play_env = SelfPlayEnv(board_shape=player.board_shape, local_player=local_player, verbose=verbose)
    game_duration_as_first = 0
    game_duration_as_second = 0
    for i in range(N):
        done = False
        board = self_play_env.reset()
        while not done:
            action = player.predict(board)
            board, reward, done, _ = self_play_env.step(action)
            
        if self_play_env.local_player_num == -1:
            wins_as_first = wins_as_first + (reward == 1)
            plays_as_first = plays_as_first + 1
            ties_as_first = ties_as_first + (reward == 0)
            game_duration_as_first = game_duration_as_first + self_play_env.n_step
        else:
            wins_as_second = wins_as_second + (reward == 1)
            ties_as_second = ties_as_second + (reward == 0)
            game_duration_as_second = game_duration_as_second + self_play_env.n_step
            plays_as_second = plays_as_second + 1
    print(f'Wins as first: {wins_as_first/(plays_as_first + eps)}')
    print(f'Wins as second: {wins_as_second/(plays_as_second + eps)}')
    print(f'Ties as first: {ties_as_first/(plays_as_first + eps)}')
    print(f'Ties as second: {ties_as_second/(plays_as_second + eps)}')
    print(f'Plays as first: {plays_as_first}')
    print(f'Plays as second: {plays_as_second}')
    print(f'Avg game duration as first: {game_duration_as_first/(N + eps)}')
    print(f'Avg game duration as second: {game_duration_as_second/(N + eps)}')
    return wins_as_first/(plays_as_first + eps), wins_as_second/(plays_as_second + eps), ties_as_first/(plays_as_first + eps), ties_as_second/(plays_as_second + eps)