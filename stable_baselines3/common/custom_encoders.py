from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp

'''
class TaskEncoder(base_component.Component):
    def __init__(
        self,
        pretrained_embedding_cfg: ConfigType,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ):
        """Encode the task into a vector.

        Args:
            pretrained_embedding_cfg (ConfigType): config for using pretrained
                embeddings.
            num_embeddings (int): number of elements in the embedding table. This is
                used if pretrained embedding is not used.
            embedding_dim (int): dimension for the embedding. This is
                used if pretrained embedding is not used.
            hidden_dim (int): dimension of the hidden layer of the trunk.
            num_layers (int): number of layers in the trunk.
            output_dim (int): output dimension of the task encoder.
        """
        super().__init__()
        if pretrained_embedding_cfg.should_use:
            with open(pretrained_embedding_cfg.path_to_load_from) as f:
                metadata = json.load(f)
            ordered_task_list = pretrained_embedding_cfg.ordered_task_list
            pretrained_embedding = torch.Tensor(
                [metadata[task] for task in ordered_task_list]
            )
            assert num_embeddings == pretrained_embedding.shape[0]
            pretrained_embedding_dim = pretrained_embedding.shape[1]
            pretrained_embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_embedding,
                freeze=True,
            )
            projection_layer = nn.Sequential(
                nn.Linear(
                    in_features=pretrained_embedding_dim, out_features=2 * embedding_dim
                ),
                nn.ReLU(),
                nn.Linear(in_features=2 * embedding_dim, out_features=embedding_dim),
                nn.ReLU(),
            )
            projection_layer.apply(agent_utils.weight_init)
            self.embedding = nn.Sequential(  # type: ignore [call-overload]
                pretrained_embedding,
                nn.ReLU(),
                projection_layer,
            )

        else:
            self.embedding = nn.Sequential(
                nn.Embedding(
                    num_embeddings=num_embeddings, embedding_dim=embedding_dim
                ),
                nn.ReLU(),
            )
            self.embedding.apply(agent_utils.weight_init)
        self.trunk = agent_utils.build_mlp(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        self.trunk.apply(agent_utils.weight_init)

    def forward(self, env_index: TensorType) -> TensorType:
        return self.trunk(self.embedding(env_index))
'''

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


class FiLM(FeedForwardEncoder):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        feature_dim: int,
        num_layers: int,
        hidden_dim: int,
        should_tie_encoders: bool,
    ):
        super().__init__(
            env_obs_shape=env_obs_shape,
            multitask_cfg=multitask_cfg,
            feature_dim=feature_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            should_tie_encoders=should_tie_encoders,
        )

        # overriding the type from base class.
        self.trunk: List[ModelType] = agent_utils.build_mlp_as_module_list(  # type: ignore[assignment]
            input_dim=env_obs_shape[0],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=feature_dim,
        )

    def forward(self, mtobs: MTObs, detach: bool = False):
        env_obs = mtobs.env_obs
        task_encoding: TensorType = cast(TensorType, mtobs.task_info.encoding)  # type: ignore[union-attr]
        # mypy raises a false alarm. mtobs.task if already checked to be not None.
        gammas_and_betas: List[TensorType] = torch.split(
            task_encoding.unsqueeze(2), split_size_or_sections=2, dim=1
        )
        # assert len(gammas_and_betas) == len(self.trunk)
        h = env_obs
        for layer, gamma_beta in zip(self.trunk, gammas_and_betas):
            h = layer(h) * gamma_beta[:, 0] + gamma_beta[:, 1]
        if detach:
            h = h.detach()

        return h