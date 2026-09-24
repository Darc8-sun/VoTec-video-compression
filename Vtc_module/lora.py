# -*- coding: utf-8 -*-

import math
import logging
from typing import List, Tuple

import torch
import torch.nn as nn


class LoRAAdapter3d(nn.Module):
    """A single 3D LoRA adapter: output += lora_B(lora_A(output)) * scaling"""

    def __init__(self, channels: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Conv3d(channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv3d(rank, channels, kernel_size=1, bias=False)
        # Initialization: A uses kaiming, B uses zero -> LoRA output is initially 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class LoRAAdapter2d(nn.Module):
    """A single 2D LoRA adapter: output += lora_B(lora_A(output)) * scaling"""

    def __init__(self, channels: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class VAEDecoderLoRA(nn.Module):
    """
    Manage LoRA injection into the WanVAE Decoder.

    Injection is done via forward hooks, without modifying the original model
    structure and without breaking the cache mechanism.

    Args:
        vae_model (WanVAE_): the internal model of WanVAE (i.e., WanVAE.model)
        rank (int): LoRA rank; larger means more expressive but more parameters
        alpha (float): LoRA scaling factor
        target_conv3d (bool): whether to inject LoRA into CausalConv3d (Conv3d) layers
        target_conv2d (bool): whether to inject LoRA into Conv2d layers

    Example:
        vae = WanVAE(z_dim=16, vae_pth="...", device="cuda")
        vae.model.eval()
        for p in vae.model.parameters():
            p.requires_grad = False

        vae_lora = VAEDecoderLoRA(vae.model, rank=8, alpha=1.0)
        vae_lora = vae_lora.to("cuda")

        # Add only the LoRA parameters to the optimizer
        optimizer = optim.AdamW(list(other_params) + list(vae_lora.parameters()), lr=1e-4)
    """

    def __init__(
        self,
        vae_model: nn.Module,
        rank: int = 8,
        alpha: float = 1.0,
        target_conv3d: bool = True,
        target_conv2d: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.adapters = nn.ModuleDict()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Collect the target layers along the decoder path
        # WanVAE_ decode path: conv2 -> decoder(conv1, middle, upsamples, head)
        target_modules = self._collect_decoder_targets(
            vae_model, target_conv3d, target_conv2d
        )

        logging.info(f"[VAEDecoderLoRA] Injecting LoRA (rank={rank}, alpha={alpha}) "
                     f"into {len(target_modules)} layers")

        for name, module in target_modules:
            out_channels = module.out_channels
            # Determine whether the layer is 3D or 2D
            if isinstance(module, nn.Conv3d):
                adapter = LoRAAdapter3d(out_channels, rank, alpha)
            elif isinstance(module, nn.Conv2d):
                adapter = LoRAAdapter2d(out_channels, rank, alpha)
            else:
                continue

            # Use a valid key name (replace '.' with '__')
            safe_name = name.replace('.', '__')
            self.adapters[safe_name] = adapter

            # Register the forward hook
            hook = module.register_forward_hook(self._make_hook(adapter))
            self._hooks.append(hook)

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"[VAEDecoderLoRA] Total LoRA parameters: {total_params:,}")

    def _collect_decoder_targets(
        self, vae_model: nn.Module, target_conv3d: bool, target_conv2d: bool
    ) -> List[Tuple[str, nn.Module]]:
        """Collect the target layers along the decoder path that need LoRA injection."""
        targets = []

        # 1. conv2 (the entry conv of the decode path)
        if target_conv3d and hasattr(vae_model, 'conv2'):
            targets.append(('conv2', vae_model.conv2))

        # 2. All Conv3d/Conv2d layers inside the decoder submodule
        if hasattr(vae_model, 'decoder'):
            decoder = vae_model.decoder
            for name, module in decoder.named_modules():
                if name == '':
                    continue
                if target_conv3d and isinstance(module, nn.Conv3d):
                    targets.append((f'decoder.{name}', module))
                elif target_conv2d and isinstance(module, nn.Conv2d) and not isinstance(module, nn.Conv3d):
                    targets.append((f'decoder.{name}', module))

        return targets

    @staticmethod
    def _make_hook(adapter: nn.Module):
        """Create the forward hook closure."""
        def hook_fn(module, input, output):
            # output may be 3D [B, C, T, H, W] or 2D [B, C, H, W]
            return output + adapter(output)
        return hook_fn

    def remove_hooks(self):
        """Remove all hooks (for cleanup at inference time)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, num_adapters={len(self.adapters)}"


class LoRAAdapterLinear(nn.Module):
    """A single Linear LoRA adapter: output += lora_B(lora_A(input)) * scaling"""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        # Initialization: A uses kaiming, B uses zero -> LoRA output is initially 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class WanFlowLoRA(nn.Module):
    """
    Manage LoRA injection into WanModel (WanFlow).

    Via forward hooks, low-rank residuals are computed from the inputs of the
    attention q/k/v/o and FFN Linear layers and added to their outputs, without
    modifying the original model structure or breaking the original computation.

    output_new = output_original + lora_B(lora_A(input)) * scaling
    Equivalent to: y = x(W + BA)^T, i.e., the standard LoRA decomposition.

    Args:
        wan_model (WanModel): the loaded and frozen WanFlow model
        rank (int): LoRA rank; larger means more expressive but more parameters
        alpha (float): LoRA scaling factor
        target_modules (list[str]): list of target layer name suffixes;
            defaults to the attention q/k/v/o and FFN layers

    Example:
        WanFlow = WanModel.from_pretrained(...)
        WanFlow.eval().requires_grad_(False)
        flow_lora = WanFlowLoRA(WanFlow, rank=8, alpha=1.0)
        flow_lora = flow_lora.to('cuda')
        # Add flow_lora.parameters() to the optimizer
    """

    # Suffixes of the Linear layers inside WanAttentionBlock that get LoRA injected
    DEFAULT_TARGETS = [
        'self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o',
        'cross_attn.q', 'cross_attn.k', 'cross_attn.v', 'cross_attn.o',
        'ffn.0', 'ffn.2',
    ]

    def __init__(
        self,
        wan_model: nn.Module,
        rank: int = 8,
        alpha: float = 1.0,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.adapters = nn.ModuleDict()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        if target_modules is None:
            target_modules = self.DEFAULT_TARGETS

        num_adapters = 0
        for block_idx, block in enumerate(wan_model.blocks):
            for name, module in block.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                # Check whether it is a target layer (exact suffix match)
                if not any(name == suffix or name.endswith('.' + suffix)
                           for suffix in target_modules):
                    continue

                adapter = LoRAAdapterLinear(
                    module.in_features, module.out_features, rank, alpha)
                # Use a valid key name
                safe_name = f'block{block_idx}__{name.replace(".", "__")}'
                self.adapters[safe_name] = adapter

                # Register the forward hook
                hook = module.register_forward_hook(self._make_hook(adapter))
                self._hooks.append(hook)
                num_adapters += 1

        logging.info(f"[WanFlowLoRA] Injecting LoRA (rank={rank}, alpha={alpha}) "
                     f"into {num_adapters} Linear layers "
                     f"across {len(wan_model.blocks)} blocks")
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"[WanFlowLoRA] Total LoRA parameters: {total_params:,}")

    @staticmethod
    def _make_hook(adapter: nn.Module):
        """Create the forward hook closure: low-rank residual from the input, added to the output."""
        def hook_fn(module, input, output):
            # input[0] is the input x of the Linear layer; adapter(x) = lora_B(lora_A(x))
            return output + adapter(input[0])
        return hook_fn

    def remove_hooks(self):
        """Remove all hooks (for cleanup at inference time)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, num_adapters={len(self.adapters)}"
