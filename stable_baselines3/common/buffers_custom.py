import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
import math

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer

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
            # get parent batch
            obs_parent_goals_ = self.obs_parent_goals[batch_inds, env_indices].copy()
            parent_rewards_ = self.parent_goal_rewards[batch_inds, env_indices].copy()
            ## Relabel (only if the parent goal is not none)
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
        
        
class SeparatePoliciesReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
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
        child_p: float = 0.2,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # datasharing
        self.child_p = child_p

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def init_datasharing(self, relations, all_models, agent_conductor):
        self.all_models = all_models
        self.valid_relations = {'parents': {}, 'children': {}}
        
        self.has_parent = False
        # set parent reward tracking
        if len(relations['parents']) > 0:
            self.parent_rewards_dict = {}
            for parent_name in relations['parents'].keys():
                assert parent_name not in self.parent_rewards_dict
                # check if the task is being trained
                if parent_name in self.all_models.keys():
                    # create reward storage
                    self.parent_rewards_dict[parent_name] = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
                    # record valid relation
                    self.valid_relations['parents'][parent_name] = relations['parents'][parent_name]
            if len(self.valid_relations['parents']) > 0:
                self.has_parent = True
                
        self.has_children = False
        # check for children
        if len(relations['children']) > 0:
            for child_name in relations['children'].keys():
                if child_name in self.all_models.keys():
                    self.valid_relations['children'][child_name] = relations['children'][child_name]
            if len(self.valid_relations['children']) > 0:
                self.has_children = True
                
        # get task proportions
        task = agent_conductor.get_task_from_name(self.task_name)
        self.child_proportions = task.get_child_proportions()
        
    def set_task_name(self, task_name):
        ''' Custom '''
        self.task_name = task_name
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        assert all([info['prev_task_name'] == self.task_name for info in infos]), "Task name mismatch"
        
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        
        # add parent rewards
        if self.has_parent:
            for parent_name in self.valid_relations['parents'].keys():
                self.parent_rewards_dict[parent_name][self.pos] = np.array([info[f'obs_{parent_name}_reward'] for info in infos]) # TODO: make sure timestep correct on these rewards

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        ''' Custom '''
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        
        # get own data
        unnormed_data = {}
        unnormed_data['obs'] = self.observations[batch_inds, env_indices, :].copy()
        if self.optimize_memory_usage:
            unnormed_data['next_obs'] = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :].copy()
        else:
            unnormed_data['next_obs'] = self.next_observations[batch_inds, env_indices, :].copy()
        unnormed_data['acts'] = self.actions[batch_inds, env_indices, :].copy()
        unnormed_data['dones'] = (self.dones[batch_inds, env_indices].copy() * (1 - self.timeouts[batch_inds, env_indices].copy()))
        unnormed_data['rewards'] = self.rewards[batch_inds, env_indices].copy()

        # child data
        if self.has_children and self.child_p > 0:
            # get child data
            child_p = self.calc_child_p()
            child_batch_size  = child_p * len(batch_inds) # TODO: Not actually taking a proportion here... 
            data_dict_list = self._get_child_data(child_batch_size)
            # add child data to parent data
            for key in data_dict_list.keys():
                data_dict_list[key].append(unnormed_data[key])
            
            # concat all together
            for key in data_dict_list.keys():
                unnormed_data[key] = np.concatenate(data_dict_list[key], axis=0)
                
        # assert all dones False
        assert np.sum(unnormed_data['dones']) == 0, "Some dones are True"
        
        # normalize and return
        data = (
            self._normalize_obs(unnormed_data['obs'], env),
            unnormed_data['acts'],
            self._normalize_obs(unnormed_data['next_obs'], env),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            unnormed_data['dones'].reshape(-1, 1),
            self._normalize_reward(unnormed_data['rewards'].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    
    def calc_child_p(self, min_p=0.05, max_p=0.5, child_p_strat='default'):
        # just use default value for now
        # TODO
        if child_p_strat == 'default':
            child_p = self.child_p
            
        elif child_p_strat == 'self_success':
            self_success = self._get_task_success_rate(self.task_name)
            child_p = max(min_p, max_p * (1 - self_success))
            assert False, "check works properly"
            
        elif child_p_strat == 'all_task_success':
            self_success = self._get_task_success_rate(self.task_name)
            children_successes = []
            for child in self.valid_relations['children'].keys():
                children_successes.append(self._get_task_success_rate(child))
            children_success = np.mean(children_successes)
            p = ((1 - self_success) + (children_success/2)) / 1.5
            child_p = max(min_p, max_p * p)
            assert False, "check works, make more sophisticated"
            
        else:
            raise NotImplementedError(f"{child_p_strat} not implemented for calc_child_p()")
        
        assert child_p >= min_p and child_p <= max_p
        return child_p
            
    
    def get_child_batch_split(self, batch_size, split_strat='even'):
        # # success rates
        # child_success_rates = {}
        # for child_name in self.valid_relations['children'].keys():
        #     child_success_rates[child_name] = self._get_task_success_rate(child_name)
        # # data sizes
        # child_data_sizes = {}
        # for child_name in self.valid_relations['children'].keys():
        #     child_data_sizes[child_name] = self.all_models[child_name].replay_buffer.size()
        # # % of task
        # child_task_proportions = self.child_proportions
        # # distance from parent
        # child_distance_from_parent = {}
        # for child_name in self.valid_relations['children'].keys():
        #     child_distance_from_parent[child_name] = self.valid_relations['children'][child_name]
        # assert False, "figure out what is simplest and best to use..."
        
        # just do even split for now
        child_split = {}
        
        if split_strat == 'even':
            child_batch_size = math.ceil(batch_size // len(self.valid_relations['children'])) # round up to boost batch size
            assert child_batch_size > 0
            for child_name in self.valid_relations['children'].keys():
                child_split[child_name] = child_batch_size
        
        else:
            raise NotImplementedError(f"{split_strat} not implemented for get_child_batch_split()")
        
        return child_split
        
    def _get_task_success_rate(self, task_name):
        return self.env.agent_conductor.get_task_epoch_success_rate(task_name) # TODO: use ema or this???
    
    def _get_child_data(self, batch_size):
        ''' Custom '''
        # decide split among children
        child_split = self.get_child_batch_split(batch_size)
        # create dict to store data
        sampled_data = {}
        for key in ['obs', 'acts', 'next_obs', 'dones', 'rewards']:
            sampled_data[key] = []
        # get data for each child
        for child_name in self.valid_relations['children'].keys():
            child_buffer = self.all_models[child_name].replay_buffer
            child_batch_size = child_split[child_name]
            child_upper_bound = child_buffer.buffer_size if child_buffer.full else child_buffer.pos
            if child_upper_bound > child_batch_size:
                # TODO: only keep sampling if child data buffer is growing in size?? (otherwize overfit to small data?)
                child_data = child_buffer.sample_unnormed_for_parent(child_batch_size, self.task_name)
                for key in child_data.keys():
                    sampled_data[key].append(child_data[key])
        assert len(sampled_data['obs']) > 0
        return sampled_data
    
    def sample_unnormed_for_parent(self, batch_size: int, parent_name: str):
        ''' Custom '''
        assert not self.optimize_memory_usage, "Not implemented for memory efficient variant"
        assert self.has_parent
        # sample batch indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        # TODO: sample using PER, or based on trajectory success rates...
        # return unnormed data
        return {
            'obs': self.observations[batch_inds, env_indices, :].copy(),
            'acts': self.actions[batch_inds, env_indices, :].copy(),
            'next_obs': self.next_observations[batch_inds, env_indices, :].copy(),
            'dones': (self.dones[batch_inds, env_indices].copy() * (1 - self.timeouts[batch_inds, env_indices].copy())),
            'rewards': self.parent_rewards_dict[parent_name][batch_inds, env_indices].copy()
        }