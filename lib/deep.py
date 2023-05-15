import enum
import math
import statistics
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.parameter import Parameter

import delu

from .sam import SAM
from .util import KWArgs, TaskType, is_oom_exception

# ======================================================================================
# >>> modules <<<
# ======================================================================================
# When an instance of ModuleSpec is a dict,
# it must contain the key "type" with a string value
ModuleSpec = Union[str, dict[str, Any], Callable[..., nn.Module]]


def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None:
    if d is None:
        d = weight.shape[-1]
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)


def make_trainable_vector(d: int) -> Parameter:
    x = torch.empty(d)
    _initialize_embeddings(x, None)
    return Parameter(x)


class OneHotEncoder(nn.Module):
    cardinalities: Tensor

    def __init__(self, cardinalities: list[int]) -> None:
        # cardinalities[i]`` is the number of unique values for the i-th categorical feature.
        super().__init__()
        self.register_buffer('cardinalities', torch.tensor(cardinalities))

    def forward(self, x: Tensor) -> Tensor:

        encoded_columns = [
            F.one_hot(x[..., column], cardinality)
            for column, cardinality in zip(range(x.shape[-1]), self.cardinalities)
        ]

        return torch.cat(encoded_columns, -1)


class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = make_trainable_vector(d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        assert x.shape[-1] == len(self.weight)
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)


class CatEmbeddings(nn.Module):
    def __init__(
        self,
        _cardinalities_and_maybe_dimensions: Union[list[int], list[tuple[int, int]]],
        d_embedding: Optional[int] = None,
        *,
        stack: bool = False,
    ) -> None:
        assert _cardinalities_and_maybe_dimensions
        spec = _cardinalities_and_maybe_dimensions
        if not (
            (isinstance(spec[0], tuple) and d_embedding is None)
            or (isinstance(spec[0], int) and d_embedding is not None)
        ):
            raise ValueError(
                'Invalid arguments. Valid combinations are:'
                ' (1) the first argument is a list of (cardinality, embedding)-tuples AND d_embedding is None'
                ' (2) the first argument is a list of cardinalities AND d_embedding is an integer'
            )
        if stack and d_embedding is None:
            raise ValueError('stack can be True only when d_embedding is not None')

        super().__init__()
        spec_ = cast(
            list[tuple[int, int]],
            spec if d_embedding is None else [(x, d_embedding) for x in spec],
        )
        self._embeddings = nn.ModuleList()
        for cardinality, d_embedding in spec_:
            self._embeddings.append(nn.Embedding(cardinality, d_embedding))
        self.stack = stack
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self._embeddings:
            _initialize_embeddings(module.weight, None)  # type: ignore[code]

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        assert x.shape[1] == len(self._embeddings)
        out = [module(column) for module, column in zip(self._embeddings, x.T)]
        return torch.stack(out, dim=1) if self.stack else torch.cat(out, dim=1)


class LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_embedding))
        self.bias = Parameter(Tensor(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class NLinear(nn.Module):
    def __init__(
        self, n_features: int, d_in: int, d_out: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_in, d_out))
        self.bias = Parameter(Tensor(n_features, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n_features):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class LREmbeddings(nn.Sequential):
    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__(LinearEmbeddings(n_features, d_embedding), nn.ReLU())


class PLREmbeddings(nn.Sequential):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                if lite
                else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )


class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: str,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = make_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    Head = nn.Linear

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_layer: int,
        activation: str,
        dropout: float,
    ) -> None:
        assert n_blocks > 0
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layer if block_i else d_in,
                    d_out=d_layer,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for block_i in range(n_blocks)
            ]
        )
        self.head = None if d_out is None else MLP.Head(d_layer, d_out)

    @property
    def d_out(self) -> int:
        return (
            self.blocks[-1].linear.out_features  # type: ignore[code]
            if self.head is None
            else self.head.out_features
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


class ResNet(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: str,
            activation: str,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = make_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = make_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        @property
        def d_out(self) -> int:
            return self.linear_second.out_features

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: str,
            activation: str,
        ) -> None:
            super().__init__()
            self.normalization = make_module(normalization, d_in)
            self.activation = make_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: str,
        activation: str,
    ) -> None:
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = (
            None
            if d_out is None
            else ResNet.Head(
                d_in=d_main,
                d_out=d_out,
                bias=True,
                normalization=normalization,
                activation=activation,
            )
        )

    @property
    def d_out(self) -> int:
        return (
            self.blocks[-1].d_out  # type: ignore[code]
            if self.head is None
            else self.head.linear.out_features
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        d_embedding: int,
        n_heads: int,
        dropout: float,
        d_key: Optional[int] = None,
        d_value: Optional[int] = None,
        share_key_query_projection: bool = False,
        bias: bool = True,
        initialization: str = 'kaiming',
        linformer_compression_ratio: Optional[float] = None,
        linformer_sharing_policy: Optional[str] = None,
        n_tokens: Optional[int] = None,
        probs_callback: Optional[Callable[[Tensor], None]] = None,
    ) -> None:
        super().__init__()
        if d_key is None:
            d_key = d_embedding
        if d_value is None:
            d_value = d_embedding
        if n_heads > 1 and any(d % n_heads != 0 for d in [d_embedding, d_key, d_value]):
            raise ValueError(
                'd_embedding, d_key and d_value must be multiples of n_heads'
            )
        if initialization not in ['kaiming', 'xavier']:
            raise ValueError('initialization must be "kaiming" or "xavier"')

        self.W_k = nn.Linear(d_embedding, d_key, bias)
        self.W_q = (
            None if share_key_query_projection else nn.Linear(d_embedding, d_key, bias)
        )
        self.W_v = nn.Linear(d_embedding, d_value, bias)
        self.W_out = nn.Linear(d_value, d_value, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self._initialization = initialization

        self.probs_callback = probs_callback

        def make_linformer_compression():
            assert (
                n_tokens and linformer_compression_ratio
            ), "Linformer error"  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(
                n_tokens, int(n_tokens * linformer_compression_ratio), bias=False
            )

        if linformer_compression_ratio is not None:
            self.linformer_key_compression = make_linformer_compression()
            self.linformer_value_compression = (
                None
                if linformer_sharing_policy == 'key-value'
                else make_linformer_compression()
            )
        else:
            self.linformer_key_compression = None
            self.linformer_value_compression = None

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.W_q, self.W_k, self.W_v]:
            if m is None:
                continue
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if self._initialization == 'xavier' and (
                m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        assert x_q.ndim == 3
        assert x_kv.ndim == 3
        assert x_q is x_kv or x_q.shape[1] == 1

        W_q = self.W_k if self.W_q is None else self.W_q
        k, v = self.W_k(x_kv), self.W_v(x_kv)
        q = k if x_q is x_kv and W_q is self.W_k else W_q(x_q)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, "Shape missmatch"

        if self.linformer_key_compression is not None:
            k = self.linformer_key_compression(k.transpose(1, 2)).transpose(1, 2)
            value_compression = (
                self.linformer_key_compression
                if self.linformer_value_compression is None
                else self.linformer_value_compression
            )
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)

        batch_size = len(x_q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = x_q.shape[1] if self.mode == 'full' else 1
        n_k_tokens = x_kv.shape[1]
        _attention_shape = (batch_size, self.n_heads, n_q_tokens, n_k_tokens)

        q = self._reshape(q)
        k = self._reshape(k)
        logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        probs = F.softmax(logits, dim=-1)
        if self.probs_callback is not None:
            with torch.no_grad():
                self.probs_callback(probs.reshape(*_attention_shape))
        probs = self.dropout(probs)

        x = probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    class Pooling(enum.Enum):
        CLS = 'cls'
        AVG = 'avg'
        FIRST_TOKEN = 'first-token'

    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_embedding: int,
            attention_n_heads: int,
            attention_dropout: float,
            attention_normalization: ModuleSpec,
            attention_residual_dropout: float,
            attention_skip_connection: bool,
            linformer_compression_ratio: Optional[float],
            linformer_sharing_policy: Optional[str],
            n_tokens: Optional[int],
            ffn_d_hidden: int,
            ffn_dropout: float,
            ffn_activation: ModuleSpec,
            ffn_normalization: ModuleSpec,
            ffn_residual_dropout: float,
            ffn_skip_connection: bool,
            prenormalization: bool,
            pooling_index: Optional[int],
        ):
            """The main building block of `Transformer`."""
            super().__init__()
            self.prenormalization = prenormalization
            self.pooling_index = pooling_index

            self.attention_normalization = make_module(
                attention_normalization, d_embedding
            )
            self.attention = MultiheadAttention(
                d_embedding=d_embedding,
                n_heads=attention_n_heads,
                dropout=attention_dropout,
                linformer_compression_ratio=linformer_compression_ratio,
                linformer_sharing_policy=linformer_sharing_policy,
                n_tokens=n_tokens,
            )
            self.attention_residual_dropout = nn.Dropout(attention_residual_dropout)
            self.attention_skip_connection = attention_skip_connection

            self.ffn_normalization = make_module(ffn_normalization, d_embedding)
            self.ffn = nn.Sequential(
                OrderedDict(
                    [
                        ('first_linear', nn.Linear(d_embedding, ffn_d_hidden)),
                        ('activation', make_module(ffn_activation)),
                        ('dropout', nn.Dropout(ffn_dropout)),
                        ('second_linear', nn.Linear(ffn_d_hidden, d_embedding)),
                    ]
                )
            )
            self.ffn_residual_dropout = nn.Dropout(ffn_residual_dropout)
            self.ffn_skip_connection = ffn_skip_connection

        def forward(self, x: Tensor) -> Tensor:
            for stage in ['attention', 'ffn']:
                normalization = getattr(self, stage + '_normalization')
                residual_dropout = getattr(self, stage + '_residual_dropout')
                skip_connection = getattr(self, stage + '_skip_connection')

                # start residual
                x_residual = x
                if self.prenormalization:
                    x_residual = normalization(x_residual)

                # apply the module
                if stage == 'attention':
                    if self.pooling_index is None:
                        x_residual = self.attention(x_residual, x_residual)
                    else:
                        pooling_slice = slice(
                            self.pooling_index, self.pooling_index + 1
                        )
                        x_residual = self.attention(
                            x_residual[:, pooling_slice], x_residual
                        )
                        x = x[:, pooling_slice]
                else:
                    x_residual = self.ffn(x_residual)

                # end residual
                x_residual = residual_dropout(x_residual)
                x = x + x_residual if skip_connection else x_residual
                if not self.prenormalization:
                    x = normalization(x)

            return x

    class Head(nn.Module):
        """The output module of `Transformer`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleSpec,
            normalization: ModuleSpec,
        ):
            super().__init__()
            self.normalization = make_module(normalization, d_in)
            self.activation = make_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_embedding: int,
        d_out: Optional[int],
        n_blocks: int,
        # attention
        attention_n_heads: int,
        attention_dropout: float,
        attention_normalization: str,
        attention_residual_dropout: float,
        # ffn
        ffn_d_hidden: Optional[int],
        ffn_d_hidden_factor: Optional[float],
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        ffn_residual_dropout: float,
        # block
        prenormalization: bool,
        first_prenormalization: bool,
        # inference
        pooling: Union[None, str, 'Transformer.Pooling'],
        # head
        head_activation: Optional[ModuleSpec],
        head_normalization: Optional[ModuleSpec],
        # linformer
        linformer_compression_ratio: Optional[float] = None,
        linformer_sharing_policy: Optional[str] = None,
        n_tokens: Optional[int] = None,
    ) -> None:
        if ffn_d_hidden is None:
            assert ffn_d_hidden_factor is not None
            ffn_d_hidden = int(d_embedding * ffn_d_hidden_factor)
        else:
            assert ffn_d_hidden_factor is None
        if pooling is not None:
            pooling = Transformer.Pooling(pooling)
        super().__init__()

        self.pooling = pooling
        self.pooling_index = None if pooling == Transformer.Pooling.AVG else 0
        self.cls_embedding = (
            CLSEmbedding(d_embedding) if pooling == Transformer.Pooling.CLS else None
        )

        # for CLS-based inference, in the last block there is no need to perform
        # computations for any token except for the CLS token
        last_block_pooling_token_only = pooling != Transformer.Pooling.AVG
        self.blocks = nn.Sequential(
            *[
                Transformer.Block(
                    d_embedding=d_embedding,
                    attention_n_heads=attention_n_heads,
                    attention_dropout=attention_dropout,
                    attention_normalization=(
                        'Identity'
                        if prenormalization
                        and block_idx == 0
                        and not first_prenormalization
                        else attention_normalization
                    ),
                    attention_residual_dropout=attention_residual_dropout,
                    attention_skip_connection=True,
                    linformer_compression_ratio=linformer_compression_ratio,
                    linformer_sharing_policy=linformer_sharing_policy,
                    n_tokens=n_tokens,
                    ffn_d_hidden=ffn_d_hidden,
                    ffn_dropout=ffn_dropout,
                    ffn_activation=ffn_activation,
                    ffn_normalization=ffn_normalization,
                    ffn_residual_dropout=ffn_residual_dropout,
                    ffn_skip_connection=True,
                    prenormalization=prenormalization,
                    pooling_index=(
                        self.pooling_index
                        if last_block_pooling_token_only and block_idx == n_blocks - 1
                        else None
                    ),
                )
                for block_idx in range(n_blocks)
            ]
        )
        self.head = (
            None
            if d_out is None
            else Transformer.Head(
                d_in=d_embedding,
                d_out=d_out,
                bias=True,
                activation=head_activation,  # type: ignore
                normalization=head_normalization if prenormalization else 'Identity',  # type: ignore
            )
        )

    @property
    def d_embedding(self) -> int:
        return self.blocks[0].ffn['first_linear'].in_features  # type: ignore[code]

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, n_tokens, d_embedding)

        if self.cls_embedding is not None:
            x = self.cls_embedding(x)
        x = self.blocks(x)
        if self.pooling == Transformer.Pooling.AVG:
            x = x.mean(1)
        else:
            assert self.pooling in (
                Transformer.Pooling.CLS,
                Transformer.Pooling.FIRST_TOKEN,
            )
            x = x[:, 0 if x.shape[1] == 1 else self.pooling_index]
        if self.head is not None:
            x = self.head(x)
        return x


_CUSTOM_MODULES = {
    cls.__name__: cls
    for cls in [
        LinearEmbeddings,
        LREmbeddings,
        PLREmbeddings,
        MLP,
        ResNet,
        Transformer,
    ]
}


def make_module(spec: ModuleSpec, *args, **kwargs) -> nn.Module:
    """
    >>> make_module('ReLU')
    >>> make_module(nn.ReLU)
    >>> make_module('Linear', 1, out_features=2)
    >>> make_module((lambda *args: nn.Linear(*args)), 1, out_features=2)
    >>> make_module({'type': 'Linear', 'in_features' 1}, out_features=2)
    """
    if isinstance(spec, str):
        Module = getattr(nn, spec, None)
        if Module is None:
            Module = _CUSTOM_MODULES[spec]
        else:
            assert spec not in _CUSTOM_MODULES
        return make_module(Module, *args, **kwargs)
    elif isinstance(spec, dict):
        assert not args
        assert not (set(spec) & set(kwargs))
        spec = spec.copy()
        return make_module(spec.pop('type'), **spec, **kwargs)
    elif callable(spec):
        return spec(*args, **kwargs)
    else:
        raise ValueError()


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_d_out(n_classes: Optional[int]) -> int:
    return 1 if n_classes is None or n_classes == 2 else n_classes


def make_named_sequential(*named_modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(list(named_modules)))


# ======================================================================================
# >>> optimization <<<
# ======================================================================================
def default_zero_weight_decay_condition(module_name, module, parameter_name, parameter):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            LinearEmbeddings,
            PeriodicEmbeddings,
        ),
    )


def split_parameters_by_weight_decay(
    model: nn.Module, zero_weight_decay_condition=default_zero_weight_decay_condition
) -> list[dict[str, Any]]:
    parameters_info = {}
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters():
            full_parameter_name = (
                f'{module_name}.{parameter_name}' if module_name else parameter_name
            )
            parameters_info.setdefault(full_parameter_name, ([], parameter))[0].append(
                zero_weight_decay_condition(
                    module_name, module, parameter_name, parameter
                )
            )
    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    for full_parameter_name, (results, parameter) in parameters_info.items():
        (params_without_wd if any(results) else params_with_wd)['params'].append(
            parameter
        )
    return [params_with_wd, params_without_wd]


def make_optimizer(
    module: nn.Module, type: str, *, sam: Optional[KWArgs] = None, **optimizer_kwargs
) -> torch.optim.Optimizer:
    Optimizer = getattr(optim, type)
    parameters = split_parameters_by_weight_decay(module)
    return (
        Optimizer(parameters, **optimizer_kwargs)
        if sam is None
        else SAM(parameters, Optimizer, **sam, **optimizer_kwargs)
    )


def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


# ======================================================================================
# >>> training <<<
# ======================================================================================
def get_loss_fn(task_type: TaskType, **kwargs) -> Callable[..., Tensor]:
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )
    return partial(loss_fn, **kwargs) if kwargs else loss_fn


def train_step(
    optimizer: optim.Optimizer,
    step_fn: Callable[..., Tensor],
    batch,
    chunk_size: int,
) -> tuple[Tensor, int]:
    if isinstance(optimizer, SAM):
        optimizer.zero_grad()
        loss = step_fn(batch)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        step_fn(batch).backward()
        optimizer.second_step(zero_grad=True)
        return loss, chunk_size

    batch_size = len(batch)
    random_state = delu.random.get_state()
    loss = None
    while chunk_size != 0:
        try:
            delu.random.set_state(random_state)
            optimizer.zero_grad()
            if batch_size <= chunk_size:
                loss = step_fn(batch)
                loss.backward()
            else:
                loss = None
                for chunk in delu.iter_batches(batch, chunk_size):
                    chunk_loss = step_fn(chunk)
                    chunk_loss = chunk_loss * (len(chunk) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            chunk_size //= 2
        else:
            break
    if not chunk_size:
        raise RuntimeError('Not enough memory even for batch_size=1')
    optimizer.step()
    return cast(Tensor, loss), chunk_size


def process_epoch_losses(losses: list[Tensor]) -> tuple[list[float], float]:
    losses_ = torch.stack(losses).tolist()
    return losses_, statistics.mean(losses_)
