import random
import sys
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from einops import rearrange, repeat
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoTokenizer

from ..falcon.modelling_RW import RWForCausalLM
from ..mpt.modeling_mpt import MPTForCausalLM
from ..mpt_redpajama.mosaic_gpt import MosaicGPT
from .configuration_otter import OtterConfig

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

import torch.distributed as dist


def master_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


# Add this line at the beginning of your script or in your main function
# dist.init_process_group(backend='nccl')

XFORMERS_AVAIL = False
XFORMERS_MSG_PRINTED = False  # Add this global variable
try:
    if not XFORMERS_MSG_PRINTED:  # Check if the message has been master_printed before
        import xformers.ops as xops
        from transformers import LlamaTokenizer
        from xformers_model import CLIPVisionModel, LlamaForCausalLM

        _xformers_version = importlib_metadata.version("xformers")
        if dist.is_initialized() and dist.get_rank() == 0:  # Check if the current process rank is 0
            master_print(f"Successfully imported xformers version {_xformers_version}")
except ImportError as e:
    if not XFORMERS_MSG_PRINTED:  # Check if the message has been master_printed before
        from transformers import (CLIPVisionModel, LlamaForCausalLM,
                                  LlamaTokenizer)

        if dist.is_initialized() and dist.get_rank() == 0:  # Check if the current process rank is 0
            master_print(f"Failed to import xformers: {e}")
            XFORMERS_AVAIL = False
            master_print("No xformers found. You are recommended to install xformers via `pip install xformers` or `conda install -c xformers xformers`")
            XFORMERS_MSG_PRINTED = True  # Set the variable to True after master_printing the message

# from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "RWForCausalLM": "transformer.h",
    "MPTForCausalLM": "transformer.blocks",
    "MosaicGPT": "transformer.blocks",
}

MODEL_CLASSES = {
    "LlamaForCausalLM": "llama",
    "OPTForCausalLM": "opt",
    "GPTJForCausalLM": "gptj",
    "GPTNeoXForCausalLM": "gpt_neox",
    "MPTForCausalLM": "mpt",
    "MosaicGPT": "mpt",
}


def _infer_decoder_layers_attr_name(model: nn.Module):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually.")


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls), {})  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def exists(val):
    return val is not None


class OtterPerceiverBlock(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8, mult: int = 4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        ff_dim = dim * mult
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(ff_dim, dim, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        residual_latents = latents
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        out = self.to_out(out) + residual_latents
        residual_out = out
        for layer in self.feed_forward:
            out = layer(out)
        return out + residual_out


class OtterPerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        # max_num_frames: int = 128,
        max_num_media: Optional[int] = None,
        max_num_frames: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if exists(max_num_frames) else None

        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(OtterPerceiverBlock(dim=dim, dim_head=dim_head, heads=heads, mult=ff_mult))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for block in self.layers:
            latents = block(x, latents)
        return self.norm(latents)


class OtterMaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: Optional[torch.BoolTensor] = None,
        attend_previous: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            attend_previous: bool
                If false, ignores immediately preceding image and starts attending when following image
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        if not XFORMERS_AVAIL:
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k = rearrange(k, "b n (h d) -> b h n d", h=h)
            v = rearrange(v, "b n (h d) -> b h n d", h=h)
            q = q * self.scale

            sim = torch.einsum("... i d, ... j d -> ... i j", q, k)
            if exists(media_locations):
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)
                media_time = torch.arange(T_img, device=x.device) + 1

                if not attend_previous:
                    text_time[~media_locations] += 1
                    # make sure max is still the number of images in the sequence
                    text_time[
                        text_time
                        > repeat(
                            torch.count_nonzero(media_locations, dim=1),
                            "b -> b i",
                            i=text_time.shape[1],
                        )
                    ] = 0

                # text time must equal media time if only attending to most immediate image
                # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
                mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

                text_to_media_mask = mask_op(
                    rearrange(text_time, "b i -> b 1 i 1"),
                    repeat(media_time, "j -> 1 1 1 (j n)", n=n),
                )
                sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

            sim = sim - sim.amax(dim=-1, keepdim=True).detach()
            attn = sim.softmax(dim=-1)

            if exists(media_locations) and self.only_attend_immediate_media:
                # any text without a preceding media needs to have attention zeroed out
                text_without_media_mask = text_time == 0
                text_without_media_mask = rearrange(text_without_media_mask, "b i -> b 1 i 1")
                attn = attn.masked_fill(text_without_media_mask, 0.0)

            out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            q = rearrange(q, "b n (h d) -> b n h d", h=h)
            k = rearrange(k, "b n (h d) -> b n h d", h=h)
            v = rearrange(v, "b n (h d) -> b n h d", h=h)
            attn_mask = None
            out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=self.scale)
        return self.to_out(out)


class OtterGatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.attn = OtterMaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * ff_mult, bias=False),
                nn.GELU(),
                nn.Linear(dim * ff_mult, dim, bias=False),
            ]
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: Optional[torch.BoolTensor] = None,
        attend_previous: bool = True,
    ) -> torch.Tensor:
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                attend_previous=attend_previous,
            )
            * self.attn_gate.tanh()
            + x
        )
        residual_x = x
        for ff in self.feed_forward:
            x = ff(x)
        x = x * self.ff_gate.tanh() + residual_x

        return x


class OtterLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer: nn.Module, decoder_layer: nn.Module):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Otter (https://github.com/dhansmair/otter-mini/)
    def condition_vis_x(self, vis_x) -> None:
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations) -> None:
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous) -> None:
        self.attend_previous = attend_previous

    def forward(
        self,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **decoder_layer_kwargs,
    ):
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
        )
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


class OtterLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_otter(
        self,
        media_token_id: int,
        vis_hidden_size: int,
        cross_attn_every_n_layers: int,
        use_media_placement_augmentation: bool,
    ):
        """
        Initialize Otter by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """

        gated_cross_attn_layers = nn.ModuleList(
            [
                OtterGatedCrossAttentionBlock(
                    dim=self.config.hidden_size,
                    dim_visual=vis_hidden_size,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self._set_decoder_layers(nn.ModuleList([OtterLayer(gated_cross_attn_layer, decoder_layer) for gated_cross_attn_layer, decoder_layer in zip(gated_cross_attn_layers, self._get_decoder_layers())]))
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.initialized_otter = True

    def forward(self, *input, **kwargs):
        """Condition the Otter layers on the media locations before forward()"""
        if not self.initialized_otter:
            raise ValueError("Otter layers are not initialized. Please call `init_otter` first.")

        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0]
        media_locations = input_ids == self.media_token_id
        # IMPORTANT: Force `attend_previous` to True when we place training data as <image>caption<|endofchunk|>
        # attend_previous = (
        #     (random.random() < 0.5) if self.use_media_placement_augmentation else False
        # )
        attend_previous = (random.random() < 0.5) if self.use_media_placement_augmentation else True
        # attend_previous = self.only_attend_previous

        if self.__class__.__name__ == "LlamaForCausalLM":
            for layer in self.get_decoder().layers:
                layer.condition_media_locations(media_locations)
                layer.condition_attend_previous(attend_previous)
        elif self.__class__.__name__ in ["MPTForCausalLM", "MosaicGPT"]:
            for layer in self.get_decoder().blocks:
                layer.condition_media_locations(media_locations)
                layer.condition_attend_previous(attend_previous)
        else:
            master_print("inavaliable text encoder")
        return super().forward(*input, **kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self) -> None:
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_attend_previous(None)


class OtterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OtterConfig
    base_model_prefix = "otter"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OtterPerceiverBlock", "CLIPEncoderLayer", "OtterLayer", "CLIPVisionModel"]

    def _init_weights(self, module):
        """Otter requires no specific initialization"""
        return super()._init_weights(module)


class OtterModel(OtterPreTrainedModel):
    config_class = OtterConfig

    def __init__(
        self,
        config: OtterConfig,
    ):
        super().__init__(config)

        ### TODO: give "LlamaForCausalLM" as the name of text_config.architectures of Llama_based flamingo
        if "llama" not in config.text_config._name_or_path:
            if config.text_config.architectures[0] == "MPTForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct")
                lang_encoder = MPTForCausalLM(config=config.text_config)
            elif config.text_config.text_config.architectures[0] == "MosaicGPT":
                text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mosaic-llama-redpajama-final-candidate")
                lang_encoder = MosaicGPT(config=config.text_config)
            elif config.text_config.architectures[0] == "RWForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("PATH-TO-YOUR-FALCON")
                lang_encoder = RWForCausalLM(config=config.text_config)
        else:
            text_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path)
            lang_encoder = LlamaForCausalLM(config=config.text_config)
        vision_encoder = CLIPVisionModel(config=config.vision_config)
        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]

        extend_instance(lang_encoder, OtterLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        if lang_encoder.__class__.__name__ == "LlamaForCausalLM":
            lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
        # use_media_placement_augmentation is strictly false for Otter model
        self.use_media_placement_augmentation = False  # config.use_media_placement_augmentation
        self.max_num_frames = config.max_num_frames if hasattr(config, "max_num_frames") else None

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = OtterPerceiverResampler(dim=self.vis_dim, max_num_frames=self.max_num_frames)

        self.lang_encoder.init_otter(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
        )

        if "lora_config" in config.__dict__:
            master_print(f"Using LoRA with config:{config.lora_config}")
            standard_modules = ["q_proj", "v_proj"]
            lang_encoder_short_name = MODEL_CLASSES[config.text_config.architectures[0]]
            model_to_lora_modules = {
                "llama": standard_modules,
                "opt": standard_modules,
                "gptj": standard_modules,
                "gpt_neox": ["query_key_value"],
                "mpt": ["Wqkv"],
            }
            lora_config = LoraConfig(
                r=config.lora_config["r"],
                lora_alpha=config.lora_config["lora_alpha"],
                lora_dropout=config.lora_config["lora_dropout"],
                task_type=TaskType.CAUSAL_LM,
                target_modules=model_to_lora_modules[lang_encoder_short_name],
            )
            self.lang_encoder = get_peft_model(self.lang_encoder, lora_config)
            self.lang_encoder.master_print_trainable_parameters()

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.lang_encoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)

    def get_image_encoder(self) -> nn.Module:
        return self.vision_encoder

    def get_lang_encoder(self) -> nn.Module:
        return self.lang_encoder

    def tie_weights(self):
        return super().tie_weights()

    def init_weights(self):
        # Freeze all parameters in vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        for name, param in self.lang_encoder.named_parameters():
            if "gated_cross_attn_layer" not in name:
                param.requires_grad = False
        # Unfreeze LM input embeddings
        self.lang_encoder.get_input_embeddings().requires_grad_(True)
        ## MPTForCausalLM is tied word embedding
        if self.lang_encoder.__class__.__name__ == "LlamaForCausalLM":
            self.lang_encoder.lm_head.requires_grad_(True)
        # assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
        # master_print model size in billions of parameters in 2 decimal places
        master_print(f"Trainable param: {(sum(p.numel() for p in self.parameters() if p.requires_grad)) / 1e9:.2f} B")

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of Otter.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (vision_x is not None) or use_cached_vision_x, "Must provide either vision_x or use_cached_vision_x to True."

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert vision_x is None, "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        vision_x = self.vision_encoder(vision_x)[0][:, 1:, :]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)


class OtterForConditionalGeneration(OtterPreTrainedModel):
    config_class = OtterConfig

    def __init__(
        self,
        config: OtterConfig,
    ):
        super().__init__(config)
        ### TODO: give "LlamaForCausalLM" as the name of text_config.architectures of Llama_based flamingo
        if "llama" not in config.text_config._name_or_path:
            if config.text_config.architectures[0] == "MPTForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("/data2/users/xushuhan/vltrojan/mosaicml/mpt-7b-instruct")
                lang_encoder = MPTForCausalLM(config=config.text_config)
            elif config.text_config.architectures[0] == "MosaicGPT":
                text_tokenizer = AutoTokenizer.from_pretrained("/data2/users/xushuhan/vltrojan/weight/mosaicml/mosaic-llama-redpajama-final-candidate")
                lang_encoder = MosaicGPT(config=config.text_config)
            elif config.text_config.architectures[0] == "RWForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("PATH-TO-YOUR-FALCON")
                lang_encoder = RWForCausalLM(config=config.text_config)
            elif config.text_config.architectures[0] == "LlamaForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path)
                lang_encoder = LlamaForCausalLM(config=config.text_config)
            else:
                import pdb

                pdb.set_trace()
        else:
            text_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path)
            lang_encoder = LlamaForCausalLM(config=config.text_config)
        vision_encoder = CLIPVisionModel(config=config.vision_config)

        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]

        extend_instance(lang_encoder, OtterLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        # if lang_encoder.__class__.__name__ == "LlamaForCausalLM":
        #     lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
        # use_media_placement_augmentation is strictly false for Otter model
        self.use_media_placement_augmentation = False  # config.use_media_placement_augmentation
        self.max_num_frames = config.max_num_frames if hasattr(config, "max_num_frames") else None

        # Informative master_print statement
        if self.max_num_frames is None or self.max_num_frames == 1:
            master_print(f"The current model version is configured for Otter-Image with max_num_frames set to {self.max_num_frames}.")
        else:
            master_print(f"The current model version is configured for Otter-Video with a maximum of {self.max_num_frames} frames.")

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = OtterPerceiverResampler(dim=self.vis_dim, max_num_frames=self.max_num_frames)

        self.lang_encoder.init_otter(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
        )

        if "lora_config" in config.__dict__:
            original_architecture_name = self.lang_encoder.__class__.__name__
            master_print(f"Using LoRA with config:{config.lora_config}")
            standard_modules = ["q_proj", "v_proj"]
            lang_encoder_short_name = MODEL_CLASSES[config.text_config.architectures[0]]
            model_to_lora_modules = {
                "llama": standard_modules,
                "opt": standard_modules,
                "gptj": standard_modules,
                "gpt_neox": ["query_key_value"],
                "mpt": ["Wqkv"],
            }
            lora_config = LoraConfig(
                r=config.lora_config["r"],
                lora_alpha=config.lora_config["lora_alpha"],
                lora_dropout=config.lora_config["lora_dropout"],
                task_type=TaskType.CAUSAL_LM,
                target_modules=model_to_lora_modules[lang_encoder_short_name],
            )
            self.lang_encoder = get_peft_model(self.lang_encoder, lora_config)
            self.lang_encoder.master_print_trainable_parameters()
            self.lang_encoder.__class__.__name__ = f"{original_architecture_name}LoRA"

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.lang_encoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)

    def get_image_encoder(self) -> nn.Module:
        return self.vision_encoder

    def get_lang_encoder(self) -> nn.Module:
        return self.lang_encoder

    def init_weights(self):
        # Freeze all parameters in self.model if train_vision_encoder is False or train_lang_encoder is False
        if not ("train_full_model" in self.config.__dict__ and self.config.train_full_model is True):
            for param in self.parameters():
                param.requires_grad = False

        # Freeze all parameters in vision encoder
        if "train_vision_encoder" in self.config.__dict__ and self.config.train_vision_encoder is True:
            master_print("Unfreeze vision encoder.")
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        if "train_lang_encoder" in self.config.__dict__ and self.config.train_lang_encoder is True:
            master_print("Unfreeze language decoder.")
            for name, param in self.lang_encoder.named_parameters():
                param.requires_grad = True

        # Freeze all parameters in vision encoder
        if "train_vision_encoder" in self.config.__dict__ and self.config.train_vision_encoder is True:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        if "train_lang_encoder" in self.config.__dict__ and self.config.train_lang_encoder is True:
            for name, param in self.lang_encoder.named_parameters():
                param.requires_grad = True

        # Freeze all parameters in vision encoder
        if "train_vision_encoder" in self.config.__dict__ and self.config.train_vision_encoder is True:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        if "train_lang_encoder" in self.config.__dict__ and self.config.train_lang_encoder is True:
            for name, param in self.lang_encoder.named_parameters():
                param.requires_grad = True

        if "lora_config" in self.config.__dict__:
            # Use another logic to unfreeze gated_cross_attn_layers and perceivers
            master_print(f"LoRA trainable param: {(sum(param.numel() for name, param in self.lang_encoder.named_parameters() if 'lora' in name)) / 1e6:.3f} M")
            for name, param in self.lang_encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        for name, param in self.lang_encoder.named_parameters():
            if "gated_cross_attn_layer" in name:
                param.requires_grad = True

        for name, param in self.named_parameters():
            if "perceiver" in name:
                param.requires_grad = True
        # Unfreeze LM input and output embeddings
        self.lang_encoder.get_input_embeddings().requires_grad_(True)
        ## MPTForCausalLM is tied word embedding
        if "LlamaForCausalLM" in self.lang_encoder.__class__.__name__:
            self.lang_encoder.lm_head.requires_grad_(True)

        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                master_print(f"Parameter: {name}, Size: {param.numel() / 1e6:.6f} M")
        master_print(f"Total Trainable param: {total_params / 1e9:.6f} B")

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        is_ccp_loss = False,
        bd_type = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of Otter.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (vision_x is not None) or use_cached_vision_x, "Must provide either vision_x or use_cached_vision_x to True."

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert vision_x is None, "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            is_ccp_loss=is_ccp_loss,
            bd_type=bd_type,
            **kwargs,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        vision_x = self.vision_encoder(vision_x)[0][:, 1:, :]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    @torch.no_grad()
    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if hasattr(self, "_hf_hook"):
            # add a hook to make sure that the output of lang_encoder is mapped to the same device as the lang_x
            hook = AlignDevicesHook(
                execution_device=lang_x.device,
                io_same_device=True,
                place_submodules=False,
            )
            add_hook_to_module(self.lang_encoder, hook)
        num_beams = generate_kwargs.get("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)
        self._encode_vision_x(vision_x=vision_x)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=self.eoc_token_id,
            **generate_kwargs,
        )

        self.lang_encoder.clear_conditioned_layers()
        return output
