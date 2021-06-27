from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gym
import numpy as np

def create_resnet_model(n_input_channels, kernel_size=3):
    model = th.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    layers = [l for l in model.children() if not isinstance(l, nn.MaxPool2d)][:-1]
    
    layers[4][0].conv1.stride = (1, 1)
    layers[4][0].downsample[0].stride = (1, 1)
    layers[5][0].conv1.stride = (1, 1)
    layers[5][0].downsample[0].stride = (1, 1)
    layers[6][0].conv1.stride = (1, 1)
    layers[6][0].downsample[0].stride = (1, 1)
    layers.append(nn.Flatten())
    new_model = th.nn.Sequential(*layers)
    new_model[0] = nn.Conv2d(n_input_channels, 64, kernel_size=(kernel_size, kernel_size), stride = (1, 1), padding = (1, 1))
    return new_model

def create_basic_cnn(n_input_channels, kernel_size = 6, padding = 3):
    cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
#             nn.Conv2d(128, 64, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
    return cnn

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
#         self.cnn = create_basic_cnn(n_input_channels, kernel_size = 6, padding = 3)
        self.cnn = create_resnet_model(n_input_channels, kernel_size=3)
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear_1 = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.linear_2 = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        return self.linear_2(self.linear_1(self.cnn(observations)))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


class ResnetCNN(th.nn.Module):
    def __init__(self, n_input_channels, features_dim, only_policy_dist=False):
        super(ResnetCNN, self).__init__()
        self.only_policy_dist = only_policy_dist
        self.cnn = create_resnet_model(n_input_channels, kernel_size=3)
        # Compute shape by doing one forward pass
        self.linear_1 = th.nn.Sequential(th.nn.Linear(512, features_dim), th.nn.BatchNorm1d(features_dim) , th.nn.ReLU())
        self.linear_2 = th.nn.Sequential(th.nn.Linear(features_dim, features_dim), th.nn.BatchNorm1d(features_dim), th.nn.ReLU())
        self.policy_net = th.nn.Linear(in_features=features_dim, out_features=16)
        self.value_net = th.nn.Linear(in_features=features_dim, out_features=1)

    def forward(self, observations):
        observations = th.from_numpy(observations).float()
        features = self.cnn(observations)
        x = self.linear_1(features)
        x = self.linear_2(x)
        if self.only_policy_dist:
            return th.nn.Softmax(-1)(self.policy_net(x))
        else:
            return self.policy_net(x), self.value_net(x)

        
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    
class NewCustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(NewCustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
#         self.cnn = create_basic_cnn(n_input_channels, kernel_size = 6, padding = 3)
        self.cnn = create_resnet_model(n_input_channels, kernel_size=4)
#         self.cnn = create_basic_cnn(n_input_channels, kernel_size=3, padding=3)
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
#         self.linear_2 = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        return self.linear(self.cnn(observations))
    
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.Tanh,
        features_extractor_kwargs=dict(features_dim=128),
        features_extractor_class=NewCustomCNN,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            features_extractor_kwargs=features_extractor_kwargs,
            features_extractor_class=features_extractor_class,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.get_valid_actions = None

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
        
    def set_get_valid_actions(self, get_valid_actions):
        self.get_valid_actions = get_valid_actions
        
    def sample_masked_actions(self, obs, distribution, deterministic=False):
        def get_mask(obs):
            masks = np.zeros((len(obs), obs.shape[-1] * obs.shape[-2]))
            for i, board in enumerate(obs):
                board = board[0].cpu().numpy()
                masks[i] = 1 - self.get_valid_actions(board)
            return th.from_numpy(masks).cuda()
        masks = get_mask(obs)
        masks[masks == 1] = -np.inf
        masked_logits = distribution.logits + masks
        if deterministic:
            return th.argmax(masked_logits, axis=1)
        return th.distributions.Categorical(logits=masked_logits).sample()
        
    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        if self.get_valid_actions:
            actions = self.sample_masked_actions(obs, distribution.distribution, deterministic=deterministic)
        else:
            actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        
        if self.get_valid_actions:
            actions = self.sample_masked_actions(observation, distribution.distribution, deterministic=deterministic)
        else:
            actions = distribution.get_actions(deterministic=deterministic)
        
        
        return actions
    
class CustomActorCriticPolicyMLP(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.Tanh,
#         features_extractor_kwargs=dict(features_dim=128),
#         features_extractor_class=CustomNetwork,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicyMLP, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
#             features_extractor_kwargs=features_extractor_kwargs,
#             features_extractor_class=features_extractor_class,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.get_valid_actions = None

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
        
    def set_get_valid_actions(self, get_valid_actions):
        self.get_valid_actions = get_valid_actions
        
    def sample_masked_actions(self, obs, distribution, deterministic=False):
        def get_mask(obs):
            masks = np.zeros((len(obs), obs.shape[-1] * obs.shape[-2]))
            for i, board in enumerate(obs):
                board = board[0].cpu().numpy()
                masks[i] = 1 - self.get_valid_actions(board)
            return th.from_numpy(masks).cuda()
        masks = get_mask(obs)
        masks[masks == 1] = -np.inf
        masked_logits = distribution.logits + masks
        if deterministic:
            return th.argmax(masked_logits, axis=1)
        return th.distributions.Categorical(logits=masked_logits).sample()
        
    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        if self.get_valid_actions:
            actions = self.sample_masked_actions(obs, distribution.distribution, deterministic=deterministic)
        else:
            actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        
        if self.get_valid_actions:
            actions = self.sample_masked_actions(observation, distribution.distribution, deterministic=deterministic)
        else:
            actions = distribution.get_actions(deterministic=deterministic)
        
        
        return actions