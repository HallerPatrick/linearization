from typing import Optional

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers import LlamaConfig, AutoConfig, AutoModel, AutoModelForCausalLM

import transformers.models.llama.modeling_llama as llama

from students.mamba2.discrete_mamba import Mixer as DiscreteMambaMixer


def _cfg(config, name, default):
    return getattr(config, name, default)


class Mamba2LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Mixer params are derived from config where available; heads follow Llama config.
        num_heads = config.num_attention_heads
        self.self_attn = DiscreteMambaMixer(
            d_model=config.hidden_size,
            d_state=_cfg(config, "mamba_state_size", 128),
            n_qk_heads=num_heads,
            n_v_heads=num_heads,
            d_conv=_cfg(config, "mamba_conv_kernel", 4),
            expand=_cfg(config, "mamba_expand", 1),
            activation=_cfg(config, "mamba_hidden_act", "silu"),
            bias=_cfg(config, "mamba_use_bias", False),
            conv_bias=_cfg(config, "mamba_use_conv_bias", True),
            chunk_size=_cfg(config, "mamba_chunk_size", 256),
            layer_idx=layer_idx,
        )
        self.mlp = llama.LlamaMLP(config)
        self.input_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_res_scale = nn.Parameter(torch.full((config.hidden_size,), 1e-3))
        # self.mlp_res_scale  = nn.Parameter(torch.full((config.hidden_size,), 1e-3))

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        if attention_mask is not None:
            x = x * attention_mask[:, :, None]
        x = self.self_attn(x, return_mixer_matrix=False)["hidden_states"]

        # LayerScale / residual scaling (init small, e.g. 1e-3)
        # shape: [d_model] so it broadcasts over [B, T, d_model]
        x = x * self.attn_res_scale

        # optional: residual dropout (if you already use it elsewhere)
        if self.training and getattr(self, "resid_dropout", None) is not None:
            x = self.resid_dropout(x)

        hidden_states = residual + x

        # --- MLP branch (pre-norm) ---
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)

        # x = x * self.mlp_res_scale
        if self.training and getattr(self, "resid_dropout", None) is not None:
            x = self.resid_dropout(x)

        hidden_states = residual + x
        return (hidden_states, None)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None]
        hidden_states = self.self_attn(hidden_states, return_mixer_matrix=False)["hidden_states"]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, None)


class LlamaMambaScaffoldConfig(LlamaConfig):
    model_type = "llama_mamba_scaffold"


class LlamaMambaScaffoldPreTrainedModel(PreTrainedModel):
    config_class = LlamaMambaScaffoldConfig

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaMambaScaffoldModel(LlamaMambaScaffoldPreTrainedModel):
    config_class = LlamaMambaScaffoldConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, getattr(config, 'pad_token_id', None))
        self.layers = nn.ModuleList(
            [Mamba2LlamaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn = layer(hidden_states, attention_mask=attention_mask)
            if output_attentions:
                all_attns += (attn,)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(i for i in [hidden_states, None, all_hidden_states, all_attns] if i is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class LlamaMambaScaffoldForCausalLM(LlamaMambaScaffoldPreTrainedModel):
    config_class = LlamaMambaScaffoldConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaMambaScaffoldModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # labels = labels.to(hidden_states.device)
            # labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            # loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


AutoConfig.register(LlamaMambaScaffoldConfig.model_type, LlamaMambaScaffoldConfig)
AutoModel.register(LlamaMambaScaffoldConfig, LlamaMambaScaffoldModel)
AutoModelForCausalLM.register(LlamaMambaScaffoldConfig, LlamaMambaScaffoldForCausalLM)

StudentConfig = LlamaMambaScaffoldConfig
StudentModel = LlamaMambaScaffoldForCausalLM
