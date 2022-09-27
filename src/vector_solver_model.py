import warnings

import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from transformers import PretrainedConfig, PreTrainedModel


class AnalogyVectorSolverConfig(PretrainedConfig):
    def __init__(
            self,
            d_model: int = 512,
            hidden_dim: int = None,
            num_layers: int = 5,
            is_abelian: bool = False,
            mse_loss: bool = True,
            use_layer_norm: bool = None,
            **kwargs
    ):
        # Base class init call
        super().__init__(**kwargs)

        # Set custom config defaults
        self.is_abelian = is_abelian
        self.num_layers = num_layers
        self.d_model = d_model
        self.mse_loss = mse_loss
        if not is_abelian and hidden_dim is not None:
            # raise Exception(f"For FF network expected hidden_dim equal to d_model, config hidden_dim=None expected, got {hidden_dim}")
            warnings.warn(f"For FF network expected hidden_dim equal to d_model, config hidden_dim=None expected, got {hidden_dim}. Ignoring.")
            self.hidden_dim = None
        elif hidden_dim is None:
            self.hidden_dim = d_model
        else:
            self.hidden_dim = hidden_dim
        if use_layer_norm is True:
            if is_abelian:
                raise Exception("Config use_layer_norm=True is only used by FF solver.")
            self.use_layer_norm = use_layer_norm
        else:
            self.use_layer_norm = False


class SolverInnerBlock(nn.Module):
    def __init__(self, dim, use_layer_norm=False):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        # if self.do_batch_norm:
        self.batch_norm1 = nn.BatchNorm1d(dim) if not use_layer_norm else nn.LayerNorm(dim)
        self.batch_norm2 = nn.BatchNorm1d(dim) if not use_layer_norm else nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.batch_norm1(nn.functional.gelu(self.l1(x)) + x)
        x_ = self.batch_norm2(nn.functional.gelu(self.l2(x_)) + x + x_)
        return x_


class AnalogyVectorSolverPretrainedModel(PreTrainedModel):
    config_class = AnalogyVectorSolverConfig
    base_model_prefix = "vector_solver"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.config.is_abelian:
                torch.nn.init.xavier_normal(module.weight)
                torch.nn.init.xavier_normal(module.bias)
            else:
                torch.nn.init.kaiming_uniform(module.weight)
                torch.nn.init.kaiming_uniform(module.bias)
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class AnalogyVectorSolverModel(AnalogyVectorSolverPretrainedModel):

    def __init__(self, config: AnalogyVectorSolverConfig, **kwargs):
        super().__init__(config, **kwargs)
        # This model could either be the Abelian Group Network from Abe et al. 2021
        # or a feedforward network with some residual connections and normalization
        if config.is_abelian is True:
            # If it's Abelian then we need to learn an invertible inner function
            def subnet_fc(dims_in, dims_out):
                return nn.Sequential(nn.Linear(dims_in, config.hidden_dim), nn.GELU(),
                                     nn.Linear(config.hidden_dim, dims_out))

            self.inner_model = Ff.SequenceINN(config.d_model)
            for k in range(config.num_layers):
                self.inner_model.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

        else:
            # Otherwise we'll use this feedforward residual block
            self.inner_model = nn.Sequential(SolverInnerBlock(config.d_model * 3, config.use_layer_norm))
            self.inner_model.append(nn.Linear(config.d_model * 3, config.d_model))
            for k in range(config.num_layers - 1):
                self.inner_model.append(SolverInnerBlock(config.d_model, config.use_layer_norm))

    def forward(self, a: torch.Tensor = None, b: torch.Tensor = None, c: torch.Tensor = None,
                add: torch.Tensor = None, sub: torch.Tensor = None,
                labels: torch.Tensor = None,
                input_ids: torch.Tensor = None  # Add this even though we don't use it to keep inputs in training...
                ):
        if add is not None or sub is not None:
            if self.config.is_abelian is False:
                raise Exception(
                    f"To perform add/subtract group operation the model must be Abelian, but config.is_abelian has value {self.config.is_abelian}")
            if a is not None or b is not None or c is not None:
                raise Exception("Cannot provide both (a,b,c) and add/sub to forward method.")
            else:
                # Phi^-1(sum_i s_i Phi(x_i)) where s_i is the sign. Proof derivable from the binary group operation
                # defined as x + y := phi^-1(phi(x) + phi(y)) and additive inverse as x^-1 := phi^-1(-phi(x))
                num_add = add.shape[-2] if add is not None else 0
                num_sub = sub.shape[-2] if sub is not None else 0
                sign = torch.ones(num_add + num_sub)
                sign[-num_sub:] *= -1
                output = self.inner_model(torch.sum(self.inner_model(
                    torch.concat(
                        [] + ([add] if add is not None else []) + ([sub] if sub is not None else []),
                        dim=-2
                    ).reshape(-1, self.config.d_model)
                )[0].reshape(-1, num_add + num_sub, self.config.d_model) * sign[..., None], dim=-2), rev=True)[0]

        else:
            if not (a is not None and b is not None and c is not None):
                raise Exception(f"Must provide a, b, c parameters together if any are provided.")
            if self.config.is_abelian:
                # TODO: See if this is generally phi^(-1)(+ or - phi(x) for all x in terms)
                batch_size = a.shape[-2]
                a, b, c = self.inner_model(
                    torch.stack([a, b, c]).reshape(batch_size * 3, self.config.d_model)
                )[0].reshape(3, batch_size, self.config.d_model)
                output = self.inner_model(c + b - a, rev=True)[0]
            else:
                output = self.inner_model(torch.concat([a, b, c], dim=-1))

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss() if self.config.mse_loss is True else nn.CosineSimilarity(dim=-1)
            loss = loss_fn(output, labels) if self.config.mse_loss is True else 1. - loss_fn(output, labels).mean()

        return loss, output
