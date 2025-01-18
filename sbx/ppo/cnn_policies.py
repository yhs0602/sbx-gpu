from dataclasses import field
from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from flax.linen.initializers import constant

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Any, Callable, Dict, Optional, Union
from flax.training.train_state import TrainState
from gymnasium import spaces

from sbx.common.policies import BaseJaxPolicy
from stable_baselines3.common.type_aliases import Schedule

tfd = tfp.distributions


class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param features_dim: Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    features_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Apply the Nature CNN architecture
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)

        # Flatten the output
        x = x.reshape((x.shape[0], -1))

        # Fully connected layer
        x = nn.Dense(self.features_dim)(x)
        x = nn.relu(x)
        return x


class CnnCritic(nn.Module):
    n_units: int = 512
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = NatureCNN(features_dim=self.n_units)(x)
        x = nn.Dense(1)(x)
        return x


class CnnActor(nn.Module):
    action_dim: int
    n_units: int = 512
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    # For Discrete, MultiDiscrete and MultiBinary actions
    num_discrete_choices: Optional[Union[int, Sequence[int]]] = None
    # For MultiDiscrete
    max_num_choices: int = 0
    split_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_std(self) -> jnp.ndarray:
        # Make it work with gSDE
        return jnp.array(0.0)

    def __post_init__(self) -> None:
        # For MultiDiscrete
        if isinstance(self.num_discrete_choices, np.ndarray):
            self.max_num_choices = max(self.num_discrete_choices)
            # np.cumsum(...) gives the correct indices at which to split the flatten logits
            self.split_indices = np.cumsum(self.num_discrete_choices[:-1])
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        x = NatureCNN(features_dim=self.n_units)(x)
        action_logits = nn.Dense(self.action_dim)(x)
        if self.num_discrete_choices is None:
            # Continuous actions
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))
        elif isinstance(self.num_discrete_choices, int):
            dist = tfd.Categorical(logits=action_logits)
        else:
            # Split action_logits = (batch_size, total_choices=sum(self.num_discrete_choices))
            action_logits = jnp.split(action_logits, self.split_indices, axis=1)
            # Pad to the maximum number of choices (required by tfp.distributions.Categorical).
            # Pad by -inf, so that the probability of these invalid actions is 0.
            logits_padded = jnp.stack(
                [
                    jnp.pad(
                        logit,
                        # logit is of shape (batch_size, n)
                        # only pad after dim=1, to max_num_choices - n
                        # pad_width=((before_dim_0, after_0), (before_dim_1, after_1))
                        pad_width=((0, 0), (0, self.max_num_choices - logit.shape[1])),
                        constant_values=-np.inf,
                    )
                    for logit in action_logits
                ],
                axis=1,
            )
            dist = tfp.distributions.Independent(
                tfp.distributions.Categorical(logits=logits_padded), reinterpreted_batch_ndims=1
            )
        return dist


class CNNPPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        log_std_init: float = 0.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {"eps": 1e-5}

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=None,  # CNN is directly integrated into the policy
            features_extractor_kwargs=None,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, max_grad_norm: float) -> jax.Array:
        key, actor_key, vf_key = jax.random.split(key, 3)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)

        obs = jnp.array([self.observation_space.sample()])

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs = {"action_dim": int(np.prod(self.action_space.shape))}
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            }
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            actor_kwargs = {
                "action_dim": int(np.sum(self.action_space.nvec)),
                "num_discrete_choices": self.action_space.nvec,
            }
        elif isinstance(self.action_space, spaces.MultiBinary):
            actor_kwargs = {
                "action_dim": 2 * self.action_space.n,
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            }
        else:
            raise NotImplementedError(f"Unsupported action space: {self.action_space}")

        self.actor = CnnActor(
            n_units=512,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            **actor_kwargs,
        )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),
                    **self.optimizer_kwargs,
                ),
            ),
        )

        self.vf = CnnCritic(n_units=512, activation_fn=self.activation_fn)

        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init(vf_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),
                    **self.optimizer_kwargs,
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.vf.apply = jax.jit(self.vf.apply)

        return key

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        self.reset_noise()
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)

    def predict_all(self, observation: np.ndarray, key: jax.Array) -> np.ndarray:
        return self._predict_all(self.actor_state, self.vf_state, observation, key)

    @staticmethod
    @jax.jit
    def _predict_all(actor_state, vf_state, observations, key):
        dist = actor_state.apply_fn(actor_state.params, observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = vf_state.apply_fn(vf_state.params, observations).flatten()
        return actions, log_probs, values

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)
