from transformers.configuration_utils import PretrainedConfig


class LlambaConfig(PretrainedConfig):
    r"""Configuration class for the CustomMamba model.

    This configuration is used to instantiate the CustomMamba model according to the specified arguments,
    defining the model architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the model.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        pad_vocab_size_multiple (`int`, *optional*, defaults to 8):
            Pad the vocabulary size up to the next multiple of this value.
        lm_head_bias (`bool`, *optional*, defaults to `False`):
            Whether the LM head includes a bias term.
        d_model (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        lm_head_prenorm (`str`, *optional*, defaults to "rms"):
            Normalization type for LM head.
        n_layer (`int`, *optional*, defaults to 32):
            Number of layers in the model.
        resid_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for residual connections.
        norm_epsilon (`float`, *optional*, defaults to 1e-5):
            Epsilon value used for normalization layers.
        mlp_cfg (`dict`, *optional*):
            Configuration for the MLP (Multi-Layer Perceptron) layer, including intermediate size, activation function, and whether to use bias.
        ssm_cfg (`dict`, *optional*):
            Configuration for the SSM (State Space Model) layer, including d_state, number of heads, expansion, and other parameters.

    """

    model_type = "llamba"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int | None = None,
        hidden_size: int | None = None,
        tie_embeddings: bool = False,
        pad_vocab_size_multiple: int = 8,
        lm_head_bias: bool = False,
        n_layer: int | None = None,
        num_hidden_layers: int | None = None,
        resid_dropout: float = 0.0,
        norm_epsilon: float = 1e-5,
        mlp_cfg: dict | None = None,
        ssm_cfg: dict | None = None,
        intermediate_size: int | None = None,
        hidden_act: str | None = None,
        mlp_bias: bool | None = None,
        num_attention_heads: int | None = None,
        **kwargs,
    ):
        rms_norm_eps = kwargs.get("rms_norm_eps", None)
        if rms_norm_eps is not None and norm_epsilon == 1e-5:
            norm_epsilon = rms_norm_eps
        super().__init__(**kwargs)

        if d_model is None:
            d_model = hidden_size or kwargs.get("hidden_size")
        if n_layer is None:
            n_layer = num_hidden_layers or kwargs.get("num_hidden_layers")
        if d_model is None:
            d_model = 576
        if n_layer is None:
            n_layer = 30

        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.lm_head_bias = lm_head_bias
        self.d_model = d_model
        self.n_layer = n_layer
        self.resid_dropout = resid_dropout
        self.norm_epsilon = norm_epsilon
        # HF compatibility
        self.hidden_size = d_model
        self.num_hidden_layers = n_layer
        if num_attention_heads is None:
            num_attention_heads = kwargs.get("num_attention_heads") or kwargs.get("n_head")
        self.num_attention_heads = num_attention_heads

        # MLP (Multi-Layer Perceptron) Config
        if mlp_cfg is None:
            if intermediate_size is None:
                intermediate_size = 14336
            if hidden_act is None:
                hidden_act = "silu"
            if mlp_bias is None:
                mlp_bias = False
            mlp_cfg = {
                "intermediate_size": intermediate_size,
                "bias": mlp_bias,
                "act_fn": hidden_act,
            }
        self.mlp_cfg = mlp_cfg

        # SSM (State Space Model) Config
        num_attention_heads = self.num_attention_heads
        if ssm_cfg is None:
            ssm_cfg = {
                "d_state": 64,
                "n_v_heads": num_attention_heads or 32,
                "n_qk_heads": num_attention_heads or 32,
                "expand": 1,
                "chunk_size": 128,
                "activation": "identity",
                "bias": False,
            }
        else:
            # If num_attention_heads was explicitly passed, override ssm_cfg values
            if num_attention_heads is not None:
                ssm_cfg["n_v_heads"] = num_attention_heads
                ssm_cfg["n_qk_heads"] = num_attention_heads
            else:
                # Ensure heads are inherited from the teacher config if not set explicitly.
                ssm_cfg.setdefault("n_v_heads", self.num_attention_heads or 32)
                ssm_cfg.setdefault("n_qk_heads", self.num_attention_heads or 32)

        self.ssm_cfg = ssm_cfg
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.time_step_min = kwargs.get("time_step_min", 0.001)
        self.time_step_max = kwargs.get("time_step_max", 0.1)
        self.time_step_floor = kwargs.get("time_step_floor", 1e-4)
