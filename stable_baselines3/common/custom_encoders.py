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

ModelType = torch.nn.Module
TensorType = torch.Tensor

####################################################################################################
# inits from: https://github.com/Nirnai/DeepRL/blob/e5d9554e6999a11f3350abfc3fa605ef3ac112d7/models/torch_utils.py#L61

def naive(m):
    if isinstance(m, nn.Linear):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / float(fan_in)), b=math.sqrt(1.0 / float(fan_in)))
        nn.init.zeros_(m.bias)

def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(m.bias)

def kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(m.bias)

####################################################################################################
# inits from mtrl

# TODO: use these inits more??

def weight_init_linear(m: ModelType):
    assert isinstance(m.weight, TensorType)
    nn.init.xavier_uniform_(m.weight)
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init_conv(m: ModelType):
    assert False
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert isinstance(m.weight, TensorType)
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)  # type: ignore[operator]
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain("relu")
    assert isinstance(m.weight, TensorType)
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weight_init_moe_layer(m: ModelType):
    assert isinstance(m.weight, TensorType)
    for i in range(m.weight.shape[0]):
        # nn.init.xavier_uniform_(m.weight[i]) # TODO: kaiming better for relu?
        nn.init.kaiming_uniform_(m.weight[i])
        # nn.init.orthogonal_(m.weight[i], gain=nn.init.calculate_gain("relu"))
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init(m: ModelType):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        weight_init_linear(m)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        weight_init_conv(m)
    elif isinstance(m, MoELinear):
        weight_init_moe_layer(m)

####################################################################################################

class TaskEncoder(nn.Module):
    # TODO: improve to account for one-hot!!!
    def __init__(self, task_dim, net_arch=[100,50], out_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(*create_mlp(input_dim=task_dim, output_dim=out_dim, net_arch=net_arch))

    def forward(self, task):
        return self.encoder(task)
        

####################################################################################################

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
        mlp_out_dim: int = 50,
        mlp_net_arch: List[int] = [50],
        fuse_method: str = 'concat',
    ) -> None:
        # TODO: weight inits?
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
                if key == 'desired_goal':
                    extractors[key] = TaskEncoder(task_dim=in_dim, out_dim=mlp_out_dim)
                else:
                    extractors[key] = nn.Sequential(*create_mlp(input_dim=in_dim, output_dim=mlp_out_dim, net_arch=mlp_net_arch)) # TODO: device, check size correct
                total_concat_size += mlp_out_dim
                print("no activation on output layer...")

            else: 
                assert False

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

############################################################################################################

class FiLM(BaseFeaturesExtractor):
    '''
    # TODO: other papers pass the language representation through a weight and bias layer...
    - https://github.com/mila-iqia/babyai/blob/65fb0cb6f816532a65014bf034a758244e2d5ae7/babyai/model.py#L20
    - https://github.com/CraftJarvis/MC-Controller/blob/68b5a9d08036301c731157e1d5f6a6f25138dd3e/src/utils/foundation.py#L65

    # TODO:
    - ensure use same params as mtrl.
    - Return task embedding or only use to modulate encoding?

    '''
    def __init__(
        self,
        # env_obs_shape: List[int],
        observation_space: spaces.Dict,
        feature_dim: int = 50,
        hidden_dim: int = 50,
        num_layers = 1,
        concat_task_encoding: bool = True,
        # num_layers: int,
        # hidden_dim: int,
        # should_tie_encoders: bool,
    ):
        # TODO: weight inits?
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)
    
        self.concat_task_encoding = concat_task_encoding

        # Task encoder
        task_dim = observation_space['desired_goal'].shape[0]
        n_gammas_betas = 2 * (num_layers + 1)
        self.film_task_encoder = TaskEncoder(task_dim=task_dim, out_dim=n_gammas_betas)

        # Obs encoder
        obs_dim = observation_space['observation'].shape[0]
        self.trunk = build_mlp_as_module_list(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=feature_dim, num_layers=num_layers)
        # TODO: should there be a relu on output of this trunk also??
        assert len(self.trunk) == num_layers + 1

        if self.concat_task_encoding:
            self._features_dim = feature_dim + n_gammas_betas
        else:
            self._features_dim = feature_dim

    def forward(self, observations: TensorDict, detach: bool = False) -> th.Tensor:
        
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
            assert False
        
        if self.concat_task_encoding:
            h = torch.cat([h, task_encoding], dim=1)
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


############################################################################################################

class MoELinear(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, bias: bool = True
    ):
        """torch.nn.Linear layer extended for use as a mixture of experts.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.rand(self.num_experts, self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_experts, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x: TensorType) -> TensorType:
        if self.use_bias:
            return x.matmul(self.weight) + self.bias
        else:
            return x.matmul(self.weight)

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features = 50,
        num_layers = 2,
        hidden_features = 50,
        bias: bool = True,
    ):
        """A feedforward model of mixture of experts layers.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            num_layers (int): number of layers in the feedforward network.
            hidden_features (int): dimensionality of hidden layer in the
                feedforward network.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        layers: List[nn.Module] = []
        current_in_features = in_features
        for _ in range(num_layers - 1):
            linear = MoELinear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layers.append(linear)
            layers.append(nn.ReLU())
            current_in_features = hidden_features
        linear = MoELinear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )
        layers.append(linear)
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)

    def __repr__(self) -> str:
        return str(self._model)


class AttentionBasedExperts(nn.Module):
    # TODO: this net works differently to how described in paper??
    # - In paper does a cosine similarity between task encoding and state embedding...
    def __init__(
        self,
        num_experts,
        embedding_dim,
        net_arch=[50,50],
        temperature=1.0,
        should_detach_task_encoding=True,
    ):
        super().__init__()
        self.temperature = temperature
        self.should_detach_task_encoding = should_detach_task_encoding # TODO: check this correct
        
        self.trunk = nn.Sequential(*create_mlp(input_dim=embedding_dim, output_dim=num_experts, net_arch=net_arch))
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, task_embedding) -> TensorType:
        emb = task_embedding
        if self.should_detach_task_encoding:
            emb = emb.detach()  # type: ignore[union-attr]
        
        output = self.trunk(emb)
        gate = self._softmax(output / self.temperature)
        if len(gate.shape) > 2:
            breakpoint()
        return gate.t().unsqueeze(2)


class MixtureofExpertsEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        num_experts: int,
        task_embedding_dim = 50,
        moe_out_dim = 50,
        detach_emb_for_selection = True,
        # device: torch.device,
    ):
        """Mixture of Experts based encoder.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            encoder_cfg (ConfigType): config for the experts in the mixture.
            task_id_to_encoder_id_cfg (ConfigType): mapping between the tasks and the encoders.
            num_experts (int): number of experts.

        TODO: task encoding slightly different. If one-hot, should use embeddings and different encoders???

        """
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)
        
        self.task_encoder = TaskEncoder(task_dim=observation_space['desired_goal'].shape[0], out_dim=task_embedding_dim)

        self.selection_network = AttentionBasedExperts(
            num_experts=num_experts,
            embedding_dim=task_embedding_dim,
            should_detach_task_encoding=detach_emb_for_selection
        )

        self.moe = MoEFeedForward(
            num_experts=num_experts,
            in_features=observation_space['observation'].shape[0],
            out_features=moe_out_dim,
        )

        self.selection_network.apply(weight_init)
        self.moe.apply(weight_init)

        self._features_dim = moe_out_dim + task_embedding_dim

    def forward(self, observations: TensorDict, detach: bool = False):
        env_obs = observations['observation']
        task_obs = observations['desired_goal']

        task_embedding = self.task_encoder(task_obs)

        encoder_mask = self.selection_network(task_embedding)
        encoding = self.moe(env_obs)
        if detach:
            encoding = encoding.detach()
            assert False
        sum_of_masked_encoding = (encoding * encoder_mask).sum(dim=0)
        sum_of_encoder_count = encoder_mask.sum(dim=0)
        encoding = sum_of_masked_encoding / sum_of_encoder_count
        # TODO: according to CARE Fig 3, there is another mlp here

        out_encoding = torch.cat([encoding, task_embedding], dim=1)
        
        return out_encoding
    
class CAREDummy(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        num_experts: int,
        moe_out_dim = 50,
        # device: torch.device,
    ):
        """Mixture of Experts based encoder.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            encoder_cfg (ConfigType): config for the experts in the mixture.
            task_id_to_encoder_id_cfg (ConfigType): mapping between the tasks and the encoders.
            num_experts (int): number of experts.

        TODO: task encoding slightly different. If one-hot, should use embeddings and different encoders???

        """
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)
        
        assert num_experts == 1
        self.moe = MoEFeedForward(
            num_experts=num_experts,
            in_features=observation_space['observation'].shape[0],
            out_features=moe_out_dim,
        )
        self.moe.apply(weight_init)
        self._features_dim = moe_out_dim

    def forward(self, observations: TensorDict, detach: bool = False):
        env_obs = observations['observation']
        encoding = self.moe(env_obs).sum(dim=0) # summed over wrong dimension???
        return encoding