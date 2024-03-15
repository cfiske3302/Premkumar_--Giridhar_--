from stable_baselines3 import PPO
from stable_baselines3.common.policies import *
from stable_baselines3.common.torch_layers import NatureCNN
from torch import nn
import gymnasium as gym


class TetisFeatureExtractorLookAhead(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        # print(observation_space)
        # assert isinstance(observation_space, spaces.Box), (
        #     "NatureCNN must be used with a gym.spaces.Box ",
        #     f"observation space, not {observation_space}",
        # )
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 20), stride=(1, 20), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flat = nn.Flatten()

        with th.no_grad():
            # print(th.as_tensor(observation_space.sample()[None]).float().shape)
            # print(type(th.as_tensor(observation_space.sample()[None]).float()))
            # print(observation_space.shape)
            observations = (
                th.as_tensor(observation_space.sample()[None])
                .float()
                .permute(0, 3, 1, 2)
            )
            # print(th.as_tensor(observation_space.sample()[None, None]).float().shape)
            # screens = th.cat(
            #     (observations[:, :, :, :10], observations[:, :, :, 17:27]), axis=-1
            # )
            # infos = self.flat(
            #     th.cat(
            #         (observations[:, :, :, 10:17], observations[:, :, :, 27:]), axis=-1
            #     )
            # )
            # print("screens", screens.shape)
            # print("self.cnn(screens)", self.cnn(screens).shape)
            # print("infos", infos.shape)
            features = self.cnn(observations)
            n_flatten = features.shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = th.as_tensor(observations).permute(0, 3, 1, 2)
        # print(observations.shape)
        # observations = observations[:, None]
        # screens = th.cat(
        #     (observations[:, :, :, :10], observations[:, :, :, 17:27]), axis=-1
        # )
        # infos = self.flat(
        #     th.cat((observations[:, :, :, 10:17], observations[:, :, :, 27:]), axis=-1)
        # )

        return self.linear(self.cnn(observations))


class TetrisActorCriticCnnPolicyLookAhead(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[
            BaseFeaturesExtractor
        ] = TetisFeatureExtractorLookAhead,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
