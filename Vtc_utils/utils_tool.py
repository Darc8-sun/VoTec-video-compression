import torch.optim as optim
'''
# --------------------------------------------------------------------------------
#   Color fixed script from Li Yi (https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
# --------------------------------------------------------------------------------
'''

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage
import torch
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x
    def sample_r(self,time):
        B,C,T,H,W=self.mean.shape
        if time is not None and not time.is_floating_point():
            time = time.float()
        time=time.view(B,1,1,1,1)
        x = self.mean + (time/1000) * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x
    def kl(self,x,time=None):
        B,C,T,H,W=x.shape
        if time is not None and not time.is_floating_point():
            time = time.float()
        time=time.view(B,1,1,1,1)
        x_t=(1-time/1000)*x
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            kl_loss=0.5 * torch.sum(
                torch.pow(self.mean - x_t, 2) / (time**2)
                + self.var / (time**2) - 1.0 - self.logvar + 2*torch.log(time),
                dim=[1, 2, 3,4])
            return kl_loss/(C*T*H*W)
    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )










def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std



class StraightThroughRound(torch.autograd.Function):
    """Straight-Through Estimator (STE) for rounding, keeps gradient flow during training."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_tensor(x: Tensor, num_bits: int = 8,
                    min_val: float = None, max_val: float = None,
                    symmetric: bool = False):
    """Uniform quantization with configurable precision.

    Quantizes a float tensor into discrete levels defined by `num_bits`,
    then dequantizes back to float for downstream use. Gradient is preserved
    via Straight-Through Estimator (STE) during training.

    Args:
        x (Tensor): Input tensor, e.g. style_mean or style_std of shape (B, C, 1, 1).
        num_bits (int): Quantization precision in bits. Number of levels = 2^num_bits.
            Default: 8.
        min_val (float | None): Lower bound of quantization range. If None, auto-computed
            from x.min(). Default: None.
        max_val (float | None): Upper bound of quantization range. If None, auto-computed
            from x.max(). Default: None.
        symmetric (bool): If True, use symmetric range [-R, R] where R = max(|min|, |max|).
            Suitable for signed values like style_mean. Default: False.

    Returns:
        dict:
            - 'x_q'     (Tensor): Dequantized float tensor (same shape as x).
            - 'x_int'   (Tensor): Quantized integer codes (same shape as x, dtype long).
            - 'scale'   (float):  Quantization step size.
            - 'min_val' (float):  Effective lower bound used.
            - 'max_val' (float):  Effective upper bound used.
            - 'num_bits' (int):   Bits used.
    """
    num_levels = 2 ** num_bits

    # --- Determine quantization range ---
    if min_val is None:
        min_val = float(x.min().item())
    if max_val is None:
        max_val = float(x.max().item())

    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        min_val, max_val = -abs_max, abs_max

    # Guard against degenerate range
    if max_val == min_val:
        max_val = min_val + 1e-6

    scale = (max_val - min_val) / (num_levels - 1)

    # --- Quantize (STE: forward = round, backward = identity) ---
    x_clamped = x.clamp(min_val, max_val)
    x_scaled  = (x_clamped - min_val) / scale
    x_int     = StraightThroughRound.apply(x_scaled).long()   # integer codes [0, num_levels-1]
    x_q       = x_int.float() * scale + min_val                # dequantize

    return {
        'x_q':      x_q,
        'x_int':    x_int,
        'scale':    scale,
        'min_val':  min_val,
        'max_val':  max_val,
        'num_bits': num_bits,
    }


def quantize_style_params(style_mean: Tensor, style_std: Tensor,
                          num_bits_mean: int = 8, num_bits_std: int = 8,
                          mean_range: tuple = None, std_range: tuple = None):
    """Convenience wrapper to quantize both style_mean and style_std together.

    style_mean is signed  → symmetric quantization by default.
    style_std  is positive → asymmetric (unsigned) quantization by default.

    Args:
        style_mean (Tensor): Shape (B, C, 1, 1), channel-wise mean.
        style_std  (Tensor): Shape (B, C, 1, 1), channel-wise std.
        num_bits_mean (int): Bits for style_mean. Default: 8.
        num_bits_std  (int): Bits for style_std.  Default: 8.
        mean_range (tuple | None): (min_val, max_val) for style_mean. None = auto.
        std_range  (tuple | None): (min_val, max_val) for style_std.  None = auto.

    Returns:
        dict:
            - 'mean' : quantize_tensor result dict for style_mean
            - 'std'  : quantize_tensor result dict for style_std
            - 'total_bits' (int): total number of quantized bits for both tensors
    """
    mean_min = mean_range[0] if mean_range else None
    mean_max = mean_range[1] if mean_range else None
    std_min  = std_range[0]  if std_range  else 0.0   # std is always >= 0
    std_max  = std_range[1]  if std_range  else None

    q_mean = quantize_tensor(style_mean, num_bits=num_bits_mean,
                             min_val=mean_min, max_val=mean_max,
                             symmetric=True)

    q_std  = quantize_tensor(style_std, num_bits=num_bits_std,
                             min_val=std_min, max_val=std_max,
                             symmetric=False)

    total_bits = style_mean.numel() * num_bits_mean + style_std.numel() * num_bits_std

    return {
        'mean':       q_mean,
        'std':        q_std,
        'total_bits': total_bits,
    }






def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor,bits=None):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)

    if bits is not None:
        style = quantize_style_params(style_mean, style_std, num_bits_mean=bits, num_bits_std=bits)
        style_mean, style_std = style['mean']['x_q'], style['std']['x_q']
        bitsforstyle = style['total_bits']
    else:
        bitsforstyle = None
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size), bitsforstyle

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq

def configure_optimizers_vq(net, learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=learning_rate, betas=(0.9, 0.999),
    )

    return optimizer, None


