import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.buffers import ReplayBuffer

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
    
    
expected_obs_keys = ['observation', 'desired_goal']

class LLMBasicReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        # custom parameters:
        keep_goals_same: bool = True,
        do_parent_relabel: bool = False,
        parent_relabel_p: float = 0.1,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        
        self.keep_goals_same = keep_goals_same
        self.do_parent_relabel = do_parent_relabel
        self.parent_relabel_p = parent_relabel_p

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        # Check only expected obs keys
        assert all([key in expected_obs_keys for key in self.obs_shape.keys()])
        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Handle parent goal info
        if do_parent_relabel:
            self.obs_parent_goals = np.zeros((self.buffer_size, self.n_envs, observation_space['desired_goal'].shape[0]), dtype=observation_space['desired_goal'].dtype)
            self.parent_goal_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            
        # record parent task info
        if self.do_parent_relabel:
            self.obs_parent_goals[self.pos] = np.array([info.get("obs_parent_goal", np.nan) for info in infos])
            self.parent_goal_rewards[self.pos] = np.array([info.get("obs_parent_goal_reward", np.nan) for info in infos])
            
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        '''
        Testing:
        - make sure buffer unchanged after sampling
        - make sure relabelling correct
        - make sure goals same
        '''
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Obs: remove extra dimension (we are using only one env for now), copy as doing relabelling
        obs_ = {key: obs[batch_inds, env_indices, :].copy() for key, obs in self.observations.items()}
        next_obs_ = {key: obs[batch_inds, env_indices, :].copy() for key, obs in self.next_observations.items()}
        
        # Rewards
        rewards_ = self.rewards[batch_inds, env_indices].copy() # (batch_size,)
        
        ###### Parent goal relabelling (custom) # TODO: move into own function?
        if self.do_parent_relabel:
            obs_parent_goals_ = self.obs_parent_goals[batch_inds, env_indices].copy()
            parent_rewards_ = self.parent_goal_rewards[batch_inds, env_indices].copy()
            # Relabel (only if the parent goal is not none)
            # choose idxs
            valid_idxs = np.where(~np.isnan(obs_parent_goals_[:, 0]))[0]
            prob_indices = np.where(np.random.rand(len(valid_idxs)) <= self.parent_relabel_p)[0]
            selected_idxs = valid_idxs[prob_indices]
            # relabel
            obs_['desired_goal'][selected_idxs] = obs_parent_goals_[selected_idxs]
            next_obs_['desired_goal'][selected_idxs] = obs_parent_goals_[selected_idxs] # TODO: "Always using obs parent goal for next obs! (for now)"
            rewards_[selected_idxs] = parent_rewards_[selected_idxs] # TODO: include child reward as a mini bonus?
            assert np.all(~np.isnan(rewards_)) and np.all(~np.isnan(obs_['desired_goal'])), "NaNs in rewards or goals!"
        
        ####### Set next_obs goal to be same as obs (custom)
        if self.keep_goals_same:
            next_obs_['desired_goal'] = obs_['desired_goal'].copy()
            
        # Normalize obs if needed
        obs_ = self._normalize_obs(obs_, env)
        next_obs_ = self._normalize_obs(next_obs_, env)

        # Convert obs to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}
        
        # Rewards process and to tensor
        rewards = self.to_torch(self._normalize_reward(rewards_.reshape(-1, 1), env))
        
        # Only use dones that are not due to timeouts
        # deactivated by default (timeouts is initialized as an array of False)
        dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
            -1, 1
        )
        assert dones.sum() == 0, "Not setup to handle llm terminations yet - assume no terminations for now"

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )