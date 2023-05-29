from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from gym import spaces
from torch import nn
import torch

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
  

class CustomSimpleCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        encoders = {'observation': 'flatten', 'desired_goal': 'flatten'},
        fuse_method: str = 'concat',
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)
        self.fuse_method = fuse_method

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            assert key in encoders.keys()
            assert not is_image_space(subspace, normalized_image=normalized_image)
            assert len(subspace.shape) == 1
            in_dim = get_flattened_obs_dim(subspace)

            if encoders[key] == 'flatten':
                extractors[key] = nn.Flatten()
                total_concat_size += in_dim
                
            elif encoders[key] == 'mlp':
                extractors[key] = nn.Sequential(*create_mlp(input_dim=in_dim, output_dim=50, net_arch=[50])) # TODO: device, check size correct
                total_concat_size += 50
                print("no activation on output layer...")

            elif encoders[key] == 'task_enc':
                # TODO: implement to be similar to SB3...
                # TODO: add more depth to task embeddings?
                assert False, "not implemented, concat_size"
                # extractors[key] = TaskEncoder()

        self.extractors = nn.ModuleDict(extractors)

        if fuse_method == 'concat':
            # Update the features dim manually
            self._features_dim = total_concat_size
        else:
            assert False

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        if self.fuse_method == 'concat':
            return th.cat(encoded_tensor_list, dim=1)
        else:
            assert False


class FiLM(BaseFeaturesExtractor):
    '''
    # TODO: other papers pass the language representation through a weight and bias layer...
    - https://github.com/mila-iqia/babyai/blob/65fb0cb6f816532a65014bf034a758244e2d5ae7/babyai/model.py#L20
    - https://github.com/CraftJarvis/MC-Controller/blob/68b5a9d08036301c731157e1d5f6a6f25138dd3e/src/utils/foundation.py#L65

    # TODO: ensure use same params as mtrl.

    '''
    def __init__(
        self,
        # env_obs_shape: List[int],
        observation_space: spaces.Dict,
        feature_dim: int = 50,
        num_layers = 1,
        concat_task_encoding: bool = False,
        # num_layers: int,
        # hidden_dim: int,
        # should_tie_encoders: bool,
    ):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)
    
        self.concat_task_encoding = concat_task_encoding

        # Task encoder
        task_dim = observation_space['desired_goal'].shape[0]
        n_gammas_betas = 2 * (num_layers + 1)
        self.film_task_encoder = nn.Sequential(*create_mlp(input_dim=task_dim, output_dim=n_gammas_betas, net_arch=[50]))

        # Obs encoder
        obs_dim = observation_space['observation'].shape[0]
        self.trunk = build_mlp_as_module_list(input_dim=obs_dim, hidden_dim=50, output_dim=feature_dim, num_layers=num_layers)
        # TODO: should there be a relu on output of this trunk also??
        assert len(self.trunk) == num_layers + 1

        if concat_task_encoding:
            raise NotImplementedError
        self._features_dim = feature_dim

    def forward(self, observations: TensorDict, detach: bool = False) -> th.Tensor:
        import pdb; pdb.set_trace()
        env_obs = observations['observation']

        task_encoding = self.film_task_encoder(observations['desired_goal'])
        # TODO: detach task_encoding??

        # mypy raises a false alarm. mtobs.task if already checked to be not None.
        gammas_and_betas: List[TensorType] = torch.split(
            task_encoding.unsqueeze(2), split_size_or_sections=2, dim=1
        )
        assert len(gammas_and_betas) == len(self.trunk)

        h = env_obs
        for layer, gamma_beta in zip(self.trunk, gammas_and_betas):
            h = layer(h) * gamma_beta[:, 0] + gamma_beta[:, 1]
        
        if detach:
            h = h.detach() # TODO: check in mtrl where should detach, implement as same...
        
        if self.concat_task_encoding:
            h = torch.cat([h, task_encoding], dim=1)
            assert False, "check concat correct"
            # TODO: check whether to also return task_encoding...
        
        assert h.shape[1] == self._features_dim

        return h
    


def _get_list_of_layers(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> List[nn.Module]:
    """Utility function to get a list of layers. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module]
    if num_layers == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    return mods

def build_mlp_as_module_list(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
):
    """Utility function to build a module list of layers. This assumes all
    the hidden layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module] = _get_list_of_layers(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    sequential_layers = []
    new_layer = []
    for index, current_layer in enumerate(mods):
        if index % 2 == 0:
            new_layer = [current_layer]
        else:
            new_layer.append(current_layer)
            sequential_layers.append(nn.Sequential(*new_layer))
    sequential_layers.append(nn.Sequential(*new_layer))
    return nn.ModuleList(sequential_layers)