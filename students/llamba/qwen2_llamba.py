import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from transformers import Qwen2Config, Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.generation import GenerationMixin
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .discrete_mamba import DiscreteMamba2
from .llamba import _copy_attn_to_discrete_mamba

logger = logging.get_logger(__name__)


class Qwen2LlambaConfig(Qwen2Config):
    model_type = "qwen2_llamba"

    def __init__(
        self,
        tie_embeddings: bool = False,
        lm_head_bias: bool = False,
        ssm_cfg: dict | None = None,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        **qwen2_kwargs,
    ):
        self.tie_embeddings = tie_embeddings
        self.lm_head_bias = lm_head_bias
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        super().__init__(**qwen2_kwargs)
        self.norm_epsilon = self.rms_norm_eps
        self.initializer_range = getattr(self, 'initializer_range', 0.02)

        if ssm_cfg is None:
            self.ssm_cfg = {
                "d_state": 64,
                "n_v_heads": self.num_attention_heads,
                "n_qk_heads": self.num_attention_heads,
                "expand": 1,
                "chunk_size": 128,
                "activation": "identity",
                "bias": False,
            }
        else:
            ssm_cfg.setdefault("n_v_heads", self.num_attention_heads)
            ssm_cfg.setdefault("n_qk_heads", self.num_attention_heads)
            self.ssm_cfg = ssm_cfg


class Qwen2LlambaPreTrainedModel(Qwen2PreTrainedModel):
    config_class = Qwen2LlambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2LlambaBlock"]

    def _init_weights(self, module):
        if any(p.device.type == "meta" for p in module.parameters(recurse=False)):
            return
        if isinstance(module, DiscreteMamba2):
            A = torch.empty(module.n_v_heads, dtype=torch.float32).uniform_(0, 16)
            with torch.no_grad():
                module.A_log.copy_(torch.log(A))
            module.A_log._no_weight_decay = True

            nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(module.n_v_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min),
            ).clamp(min=self.config.time_step_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

            dt_offset = 2 * module.d_inner + 2 * module.n_qk_heads * module.d_state
            with torch.no_grad():
                module.in_proj.weight[dt_offset:, :].zero_()

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2LlambaBlock(nn.Module):
    def __init__(self, config: Qwen2LlambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.mixer = DiscreteMamba2(
            d_model=config.hidden_size,
            layer_idx=layer_idx,
            **config.ssm_cfg,
        )

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        attention_mask=None,
        output_attentions=None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        mixer_outputs = self.mixer(
            hidden_states,
            inference_params=inference_params,
        )

        if output_attentions:
            return mixer_outputs["hidden_states"],

        hidden_states = mixer_outputs["hidden_states"].to(residual.dtype) + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states,


class Qwen2LlambaModel(Qwen2LlambaPreTrainedModel):
    def __init__(self, config: Qwen2LlambaConfig):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [Qwen2LlambaBlock(config=config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        self.final_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, = layer(
                hidden_states,
                inference_params=inference_params,
                attention_mask=attention_mask,
            )

        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class Qwen2LlambaForCausalLM(Qwen2LlambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen2LlambaConfig, **kwargs):
        super().__init__(config)
        self.model = Qwen2LlambaModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.lm_head_bias
        )
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def tie_weights(self, **kwargs):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.model.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inference_params=inference_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, inference_params=None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        if inference_params is not None:
            if inference_params.seqlen_offset > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and inference_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "inference_params": inference_params,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def copy_from_teacher(self, teacher, copy_qkv=True):
        def _copy_param(dst, src):
            if dst.shape != src.shape:
                logger.warning(f"Shape mismatch: {dst.shape} vs {src.shape}")
                return False
            dst.data.copy_(src.data)
            return True

        with torch.no_grad():
            if hasattr(teacher, "get_input_embeddings"):
                t_embed = teacher.get_input_embeddings().weight
                _copy_param(self.model.embed_tokens.weight, t_embed)

            if hasattr(teacher, "model") and hasattr(teacher.model, "norm"):
                _copy_param(self.model.final_layernorm.weight, teacher.model.norm.weight)

            if not self.config.tie_embeddings and hasattr(teacher, "get_output_embeddings"):
                t_lm = teacher.get_output_embeddings().weight
                _copy_param(self.lm_head.weight, t_lm)

            if hasattr(teacher, "model") and hasattr(teacher.model, "layers"):
                for i, teacher_layer in enumerate(teacher.model.layers):
                    if i >= len(self.model.layers):
                        break
                    self_layer = self.model.layers[i]
                    if hasattr(teacher_layer, "mlp"):
                        self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict(), strict=False)
                    if hasattr(teacher_layer, "input_layernorm"):
                        self_layer.input_layernorm.load_state_dict(
                            teacher_layer.input_layernorm.state_dict(), strict=False
                        )
                    if hasattr(teacher_layer, "post_attention_layernorm"):
                        self_layer.post_attention_layernorm.load_state_dict(
                            teacher_layer.post_attention_layernorm.state_dict(), strict=False
                        )

                    if copy_qkv and hasattr(teacher_layer, "self_attn"):
                        _copy_attn_to_discrete_mamba(self_layer.mixer, teacher_layer.self_attn)


from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register(Qwen2LlambaConfig.model_type, Qwen2LlambaConfig)
AutoModelForCausalLM.register(Qwen2LlambaConfig, Qwen2LlambaForCausalLM)

StudentConfig = Qwen2LlambaConfig
StudentModel = Qwen2LlambaForCausalLM
