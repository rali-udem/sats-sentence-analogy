from typing import Optional, Tuple, Union, List

import torch
from transformers import T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from .vector_solver_model import AnalogyVectorSolverModel, AnalogyVectorSolverConfig
from .tsdae_t5 import TSDAET5ForConditionalGeneration

# TODO:
#  - TSDAESolver wrapper model
#  - Holds a TSDAET5ForConditionalGeneration model (or just T5...Model with manual pooling)
#  - Also holds a SolverModel which takes three pooled vectors and outputs another
#  - If possible, initialize T5 parameters using .from_pretrained

"""
class Config subclass T5Config
    - add fields from VectorSolverConfig manually in the constructor
    
class Model subclass TSDAE
    - init attribute module for vector solver
    - __init__ calls super().__init__
    - __init__ constructs .solver from config
    - forward batch-calls encoder on each input (a,b,c) 
    - pools and runs through vector solver model
    - calls super forward with encoding provided

"""


class T5SolverConfig(T5Config):
    def __init__(self,
                 num_solver_layers: int = 5,
                 solver_type: str = "arithmetic",
                 sequentially_encode_inputs: bool = False,
                 **kwargs
                 ):
        # Base class init call
        super().__init__(**kwargs)

        valid_solver_types = ("mean", "arithmetic", "ff", "abelian")
        if solver_type not in valid_solver_types:
            raise Exception(f"Config field solver_type must be one of {valid_solver_types}, got {solver_type}")

        # Set custom config defaults
        self.num_solver_layers = num_solver_layers
        self.solver_type = solver_type
        self.sequentially_encode_inputs = sequentially_encode_inputs


class TSDAET5SolverModel(TSDAET5ForConditionalGeneration):
    def __init__(self, config: T5SolverConfig):
        super().__init__(config)
        if config.solver_type in ("ff", "abelian"):
            solver_config = AnalogyVectorSolverConfig(
                d_model=config.d_model,
                num_layers=config.num_solver_layers,
                is_abelian=(config.solver_type == "abelian")
            )
            self.solver = AnalogyVectorSolverModel(solver_config)

    def forward(
            self,
            a: Optional[Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]] = None,
            b: Optional[Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]] = None,
            c: Optional[Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]] = None,
            add: Optional[Union[
                List[Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]],
                Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]
            ]] = None,
            sub: Optional[Union[
                List[Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]],
                Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]
            ]] = None,
            labels: Optional[Tuple[Optional[torch.LongTensor], Optional[torch.FloatTensor]]] = None,
            # Will later only want to return the pooled vector
            encode_only: Optional[bool] = None,
            solve_only: Optional[bool] = None,
            # Might run into OoM errors if we do encoder(stack(a,b,c))
            sequentially_encode_inputs: Optional[bool] = None,
            # ---- Usual parameters below!
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            # labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput, BaseModelOutput]:

        self.config: T5SolverConfig
        if sequentially_encode_inputs is None:
            sequentially_encode_inputs = self.config.sequentially_encode_inputs

        has_add = add is not None
        has_sub = sub is not None
        has_abc = a is not None or b is not None or c is not None
        has_all_abc = a is not None and b is not None and c is not None
        has_regular_inputs = input_ids is not None or encoder_outputs is not None or inputs_embeds is not None
        is_arithmetic_solver = self.config.solver_type == 'arithmetic'
        is_mean_solver = self.config.solver_type == 'mean'
        # is_ff_solver = self.config.solver_type == 'ff'

        # ENCODING
        # This branch is only for AGN solver
        if has_add or has_sub:
            raise NotImplementedError("Don't use add/sub yet.")
            if has_abc:
                raise Exception("Cannot provide both (a,b,c) and add/sub to forward method.")
            if is_ff_solver:
                raise Exception(
                    f"To perform add/subtract group operation the model must not be feedforward, but config.solver_type has value {self.config.solver_type}")
            # Encode inputs one at a time (add/sub are lists of tuples of token ids/attention tensors)
            if sequentially_encode_inputs:
                add = torch.stack([super().forward(
                    input_ids=x[0],
                    attention_mask=x[1],
                    encode_only=True
                ).last_hidden_state for x in add]) if has_add else None
                sub = torch.stack([super().forward(
                    input_ids=x[0],
                    attention_mask=x[1],
                    encode_only=True
                ).last_hidden_state for x in sub]) if has_sub else None
            # If we're not sequentially encoding (add/sub are a tuple of tensors of batched padded token ids/attentions)
            else:
                add = super().forward(input_ids=add[0], attention_mask=add[1], encode_only=True).last_hidden_state if has_add else None
                sub = super().forward(input_ids=sub[0], attention_mask=sub[1], encode_only=True).last_hidden_state if has_sub else None
            if is_arithmetic_solver:
                _pred = ((torch.sum(add, dim=0) if add is not None else 0.)
                         - (torch.sum(sub, dim=0) if sub is not None else 0.))
                solver_output = (
                    torch.nn.functional.mse_loss(_pred, labels) if encode_only and labels is not None else None,
                    _pred
                )
            else:
                solver_output = self.solver(add=add, sub=sub, labels=labels if encode_only else None)
        # This branch is for any (a,b,c)->d solver
        elif has_abc:
            if not has_all_abc:
                raise Exception(f"Must provide a, b, c parameters together if any are provided.")

            if not solve_only:
                if sequentially_encode_inputs:
                    a, b, c = (TSDAET5ForConditionalGeneration.forward(
                        self,  # super().forward(
                        input_ids=x[0],
                        attention_mask=x[1],
                        encode_only=True
                    ).last_hidden_state[:, 0, :] for x in (a, b, c))
                else:
                    a, b, c = (TSDAET5ForConditionalGeneration.forward(
                        self,
                        input_ids=torch.stack([x[0] for x in (a, b, c)]).reshape(-1, a[0].shape[-1]),
                        attention_mask=torch.stack([x[1] for x in (a, b, c)]).reshape(-1, a[1].shape[-1]),
                        encode_only=True
                    ).last_hidden_state).reshape(3, -1, self.config.d_model) #a[1].shape[-1])

            if is_arithmetic_solver:
                _pred = c + b - a
                solver_output = (
                    torch.nn.functional.mse_loss(_pred, labels) if encode_only and labels is not None else None,
                    _pred
                )
            elif is_mean_solver:
                _pred = torch.mean(torch.stack([a, b, c], dim=0), dim=0)
                solver_output = (
                    torch.nn.functional.mse_loss(_pred, labels) if encode_only and labels is not None else None,
                    _pred
                )
            else:
                solver_output = self.solver(a=a, b=b, c=c, labels=labels if encode_only else None)
        elif has_regular_inputs:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                encode_only=encode_only
            )
        else:
            raise Exception(f"Expected either all of arguments (add, sub) or (a, b, c), or any one of (input_ids, encoder_outputs, inputs_embeds).")

        if encode_only:
            return solver_output

        return super().forward(
            encoder_outputs=(solver_output[1][:, None, :],),
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_head_mask=decoder_head_mask,
            decoder_inputs_embeds=decoder_inputs_embeds
        )
