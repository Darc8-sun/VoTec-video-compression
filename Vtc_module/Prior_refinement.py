import math

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from wan.modules.attention import flash_attention
from wan.modules.model import WAN_CROSSATTENTION_CLASSES, WanAttentionBlock, Head, MLPProj, rope_params,WanI2VCrossAttention,WanSelfAttention,WanLayerNorm,sinusoidal_embedding_1d
from wan.modules.vae import ResidualBlock,RMS_norm,CausalConv3d

@torch.no_grad()
class PriorMixFomer(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))


    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        # self-attention
        device = x.device
        y = self.self_attn(
            self.norm1(x).float(), seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32,device_type=device.type):
            x = x + y
        y = self.ffn(self.norm2(x).float())
        with amp.autocast(dtype=torch.float32,device_type=device.type):
            x = x + y
        return x



class CrossFusionFomer(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.cross_attn = WanI2VCrossAttention(dim,num_heads,(-1, -1),qk_norm,eps)
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        device = x.device
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32,device_type=device.type):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32,device_type=device.type):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32,device_type=device.type):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class RRDB3D(nn.Module):
    def __init__(self, dim, out_dim, dropout):
        super().__init__()
        self.resblocks_1=  ResidualBlock(dim, out_dim, dropout)
        self.resblocks_2=  ResidualBlock(out_dim, out_dim, dropout)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
    def forward(self,x_c,x_p):
        x_p=self.resblocks_1(x_p)
        x_c+=x_p
        x_c=self.resblocks_2(x_c)
        return self.head(x_c)

class Prior_refine_DIT(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 patch_size_S=(1, 2, 2),
                 patch_size_L=(1, 8, 8),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()


        self.patch_size_S = patch_size_S
        self.patch_size_L=patch_size_L
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding_1 = nn.Conv3d(
            in_dim*2, dim, kernel_size=patch_size_S, stride=patch_size_S)

        self.patch_embedding_2 = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size_L, stride=patch_size_L)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.PriorMixBlock=nn.ModuleList([
            PriorMixFomer( dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(2)
        ])


        self.PriorCrossBlock=nn.ModuleList([
            CrossFusionFomer( dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(2)
        ])


        # head
        self.head = Head(dim, out_dim, patch_size_S, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        self.ResRefineBlock=RRDB3D(out_dim, out_dim, dropout=0.0)
        # initialize weights
        self.init_weights()

    def forward(self, x_c,x_p,t,seq_len,context,x_cache=None):
        if x_cache is None:
            x_cache=x_c
        device = self.patch_embedding_1.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        x_p=[torch.cat([u, v], dim=0) for u, v in zip(x_cache, x_p)]
        x_p = [self.patch_embedding_1(u.unsqueeze(0)) for u in x_p]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_p])
        x_p = [u.flatten(2).transpose(1, 2) for u in x_p]
        seq_lens = torch.tensor([u.size(1) for u in x_p], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x_p = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x_p
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32,device_type=self.device.type):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        kwargs_all = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,)

        for block in self.PriorMixBlock:
            x_p = block(x_p,**kwargs_all)
        x_c_c=x_c
        x_c = [self.patch_embedding_2(u.unsqueeze(0)) for u in x_c]
        # grid_sizes = torch.stack(
        #     [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_c])
        x_c = [u.flatten(2).transpose(1, 2) for u in x_c]
        # seq_lens = torch.tensor([u.size(1) for u in x_c], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        x_c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x_c
        ])
        context = torch.concat([x_c, context], dim=1)
        for block in self.PriorCrossBlock:
            x_p = block(x_p,e0
                        ,context=context,context_lens=context_lens,**kwargs_all)

        x = self.head(x_p, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x=torch.cat([u.unsqueeze(0) for u in x],dim=0)
        x_c=torch.cat([u.unsqueeze(0) for u in x_c_c],dim=0)
        x= self.ResRefineBlock(x.float(), x_c.float())
        return x

    def forward_batch(self, x_c, x_p, t, seq_len, context, x_cache=None):
        r"""
        Batch 并行的 forward。

        与 `forward` 不同，本函数直接接收 5D Tensor，不拆 list、不逐个样本循环。

        Args:
            x_c (Tensor):
                原始/参考 latent，形状 [B, C_in, F, H, W]
            x_p (Tensor):
                预测 latent，形状 [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps 张量，形状 [B]
            seq_len (int):
                最大序列长度
            context (Tensor):
                文本编码张量，形状 [B, L, C]
            x_cache (Tensor, *optional*):
                用于和 x_p 拼接的条件 latent，形状 [B, C_in, F, H, W]，默认使用 x_c

        Returns:
            Tensor:
                Batch 去噪后的视频张量，形状 [B, C_out, F, H', W']
        """
        if x_cache is None:
            x_cache = x_c

        assert x_c.dim() == 5 and x_p.dim() == 5, \
            f"forward_batch expects 5D inputs, got x_c={x_c.shape}, x_p={x_p.shape}"

        device = self.patch_embedding_1.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # ---- x_p branch (small patch) ----
        x_p_in = torch.cat([x_cache, x_p], dim=1)  # [B, 2*C_in, F, H, W]
        x_p = self.patch_embedding_1(x_p_in)  # [B, dim, F_p, H_p, W_p]
        grid_sizes = torch.tensor(
            [x_p.shape[2:]], dtype=torch.long, device=device).expand(x_p.size(0), -1)
        x_p = x_p.flatten(2).transpose(1, 2)  # [B, N, dim]

        seq_lens = torch.full(
            (x_p.size(0),), x_p.size(1), dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len
        if x_p.size(1) < seq_len:
            x_p = torch.cat([
                x_p,
                x_p.new_zeros(x_p.size(0), seq_len - x_p.size(1), x_p.size(2))
            ], dim=1)

        # time embeddings
        with amp.autocast(dtype=torch.float32, device_type=device.type):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        assert isinstance(context, torch.Tensor), \
            "forward_batch expects context as a Tensor of shape [B, L, C]"
        assert context.dim() == 3 and context.size(0) == x_c.size(0), \
            f"context must be [B,L,C], got {context.shape}"
        context_lens = None
        if context.size(1) < self.text_len:
            context = torch.cat([
                context,
                context.new_zeros(
                    context.size(0), self.text_len - context.size(1), context.size(2))
            ], dim=1)
        elif context.size(1) > self.text_len:
            context = context[:, :self.text_len]
        context = self.text_embedding(context)

        kwargs_all = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
        )

        # PriorMixBlock: self-attention on x_p
        for block in self.PriorMixBlock:
            x_p = block(x_p, **kwargs_all)

        # ---- x_c branch (large patch, used as additional context) ----
        x_c_c = x_c
        x_c = self.patch_embedding_2(x_c)  # [B, dim, F_p, H_p, W_p]
        x_c = x_c.flatten(2).transpose(1, 2)  # [B, N, dim]
        assert x_c.size(1) <= seq_len, \
            f"x_c patch tokens {x_c.size(1)} exceed seq_len {seq_len}"
        if x_c.size(1) < seq_len:
            x_c = torch.cat([
                x_c,
                x_c.new_zeros(x_c.size(0), seq_len - x_c.size(1), x_c.size(2))
            ], dim=1)

        context = torch.concat([x_c, context], dim=1)

        # PriorCrossBlock: cross-attention with context
        for block in self.PriorCrossBlock:
            x_p = block(x_p, e0, context=context, context_lens=context_lens,
                        **kwargs_all)

        x = self.head(x_p, e)
        x = self.unpatchify_batch(x, grid_sizes)
        x = self.ResRefineBlock(x.float(), x_c_c.float())
        return x

    def unpatchify_batch(self, x, grid_sizes):
        r"""
        与 unpatchify 对应，但直接处理 batch 张量。

        Args:
            x (Tensor): [B, L, C_out * prod(patch_size_S)]
            grid_sizes (Tensor): [B, 3]，batch 内所有样本网格尺寸必须相同

        Returns:
            Tensor: [B, C_out, F, H', W']
        """
        c = self.out_dim
        v = grid_sizes[0].tolist()
        actual_len = math.prod(v)

        x = x[:, :actual_len].view(x.size(0), *v, *self.patch_size_S, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(x.size(0), c, *[i * j for i, j in zip(v, self.patch_size_S)])
        return x

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding_1.weight.flatten(1))
        nn.init.xavier_uniform_(self.patch_embedding_2.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
    def unpatchify(self,x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size_S, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size_S)])
            out.append(u)
        return out