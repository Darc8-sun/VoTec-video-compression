# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import numpy as np


from compressai.ops import quantize_ste
from compressai.models.google import CompressionModel, GaussianConditional
from Vtc_compressor.ckbd import *
from Vtc_compressor.compressor_module import *
from Vtc_compressor.module.layers import DepthConvBlock2, ResidualBlockUpsample, DepthConvBlock, SubpelConv2x, ResidualBlockWithStride
from Vtc_compressor.module.stream_helper import get_padding_size,get_state_dict
import math
class RefFrame():
    def __init__(self):
        self.frame = None
        self.feature = None
        self.poc = None

class FeatureExtractor(nn.Module):
    def __init__(self,g_ch_d):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x):
        ctx_t = self.conv1(x)
        ctx = self.conv2(ctx_t)
        return ctx, ctx_t



class I_Encoder(nn.Module):
    def __init__(self,latent_ch,g_ch_enc_dec,g_ch_y):
        super().__init__()
        self.enc = nn.Sequential(
            DepthConvBlock2(latent_ch, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, g_ch_y, 3, stride=2, padding=1),
        )
    def forward(self, feature):
        feature = self.enc(feature)
        return feature




class I_Decoder(nn.Module):
    def __init__(self,latent_ch,g_ch_enc_dec,g_ch_y):
        super().__init__()
        self.dec = nn.Sequential(
            ResidualBlockUpsample(g_ch_y, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock2(g_ch_enc_dec, latent_ch)
        )
    def forward(self, feature):
        feature = self.dec(feature)
        return feature


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through_if = (x >= bound) | (grad_output < 0)
        return pass_through_if * grad_output, None


class LowerBound_sig(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """
    bound: torch.Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class Encoder(nn.Module):
    def __init__(self,latent_ch,g_ch_d,g_ch_y):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_ch, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = nn.Sequential(DepthConvBlock(g_ch_d, g_ch_d),DepthConvBlock(g_ch_d, g_ch_d))
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        self.fuse_conv1_flag = False

    def forward(self, feature, ctx):
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature
        feature = self.down(feature)
        return feature


class Decoder(nn.Module):
    def __init__(self,latent_ch,g_ch_d,g_ch_y):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, latent_ch, 1)

    def forward(self, x, ctx ):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        return feature

#
# 输入 x: [1, 4, 3, 256, 256]
#        │
#        ▼ (逐帧处理)
# x_slice: [1, 4, 256, 256]
#        │
#        ├──► I帧 (i=0) ───────────────────────────┐
#        │    enc_i: [1,4,256,256] → [1,256,64,64] │
#        │    hyper_enc_i: → [1,128,16,16]         │
#        │    hyper_dec_i: → [1,256,64,64]         │
#        │    dec_i: [1,256,64,64] → [1,4,256,256]
#             x_hat  [1, 4, 256, 256]│
#        │    x_hat → dpb[0]                       │
#        │                                          │
#        └──► P帧 (i>0) ───────────────────────────┤
#             feature_adaptor: dpb[0] → [1,256,256,256]
#             feature_extractor: ctx=[1,256,256,256], ctx_t=[1,256,256,256]
#             enc_p: [1,4,256,256]+ctx → [1,128,128,128]
#             hyper_enc_p: → [1,128,32,32]
#             temporal_prior_encoder: ctx_t → [1,128,128,128]
#             dec_p: [1,128,128,128]+ctx → [1,4,256,256]
#             x_hat → dpb[0]
class I_latent_C(CompressionModel):
    def __init__(self, N=256,latent_ch=4,y_ch=128, num_slice=5, channel_c=[16,16,32,64,128], inplace=False,cb_size=1024):
        super().__init__()
        ####################i_latent_module
        self.slice_num = num_slice
        self.slice_ch = channel_c
        self.enc_i= I_Encoder(latent_ch, 368, N)
        self.dec_i= I_Decoder(latent_ch, 368, N)
        self.hyper_enc_i = nn.Sequential(
            DepthConvBlock2(N, y_ch, inplace=inplace),
            nn.Conv2d(y_ch, y_ch, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(y_ch, y_ch, 3, stride=2, padding=1),
        )
        self.y_prior_fusion= nn.Sequential(
            DepthConvBlock2(N, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
        )
        self.hyper_dec_i = nn.Sequential(
            ResidualBlockUpsample(y_ch, y_ch, 2, inplace=inplace),
            ResidualBlockUpsample(y_ch, y_ch, 2, inplace=inplace),
            DepthConvBlock2(y_ch, N),
        )
        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(channel_c[:i]), out_dim=channel_c[i] * 2) if i else None
            for i in range(num_slice)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
                EntropyParametersEX(in_dim=N * 2 + channel_c[i] * 2, out_dim=channel_c[i] * 2)
            if i else EntropyParametersEX(in_dim=N * 2, out_dim=channel_c[i] * 2)
            for i in range(num_slice)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=N * 2 + channel_c[i] * 4, out_dim=channel_c[i] * 2)
            if i else EntropyParametersEX(in_dim=N * 2 + channel_c[i] * 2, out_dim=channel_c[i] * 2)
            for i in range(num_slice)
        )
        self.local_context = nn.ModuleList(
            nn.Conv2d(in_channels=channel_c[i], out_channels=channel_c[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(channel_c))
        )
        self.codebook_size = cb_size
        self.quantize = VectorQuantiser(self.codebook_size, y_ch, contras_loss=True)
        self.gaussian_conditional = GaussianConditional(None)
    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)
    def calculate_bpp(self,likelihood,num_pixels):

        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in likelihood
        )
        return bpp_loss
    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)
    def forward(self, x):
        y = self.enc_i(x)
        N, _, H, W = x.size()
        num_pixels = N * H * W * 8 * 8 * 1

        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc_i(y_pad)
        z_hat, emb_loss, _ = self.quantize(z)

        num_bits_z = np.log2(self.codebook_size)*z_hat.shape[-2]*z_hat.shape[-1]*z_hat.shape[0]
        z_likelihoods=torch.tensor(num_bits_z,device=z_hat.device)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        q_likelihoods = []
        params = self.hyper_dec_i(z_hat)
        ref_parms=params
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor

                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor

                # Non-anchor
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, channel_ctx, params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        q_likelihoods = torch.cat(q_likelihoods, dim=1)

        bpp_y = self.calculate_bpp(y_likelihoods,num_pixels)
        bpp_z = z_likelihoods/num_pixels
        x_hat = self.dec_i(y_hat)
        bpp=bpp_y+bpp_z
        bits = torch.sum(bpp_y + bpp_z) * num_pixels
        return_dict={
        "x_hat": x_hat,
        "bit": bits,
        "bpp": bpp,
        "bpp_y": bpp_y,
        "bpp_z": bpp_z,
        'codebook_loss': emb_loss,
         'ref_params': ref_parms,
        }
        return return_dict

    # @staticmethod
    # def get_q_scales_from_ckpt(ckpt_path):
    #     ckpt = get_state_dict(ckpt_path)
    #     q_scale_enc = ckpt["q_scale_enc"].reshape(-1)
    #     q_scale_dec = ckpt["q_scale_dec"].reshape(-1)
    #     return q_scale_enc, q_scale_dec

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)




class P_latent_C(CompressionModel):
    def __init__(self, N=256,latent_ch=4,y_ch=128, num_slice=3, channel_c=[16,48,192],cb_size=1024,inplace=False):
        super().__init__()
        self.slice_num = num_slice
        self.slice_ch = channel_c
        ####################p_latent_module
        self.enc_p = Encoder(latent_ch,N,N)
        self.dec_p= Decoder(latent_ch,N,N)
        self.hyper_enc_p =nn.Sequential(
            DepthConvBlock(N, y_ch, inplace=inplace),
            nn.Conv2d(y_ch, y_ch, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(y_ch, y_ch, 3, stride=2, padding=1),
        )
        self.hyper_dec_p = nn.Sequential(
            ResidualBlockUpsample(y_ch, y_ch, 2, inplace=inplace),
            ResidualBlockUpsample(y_ch, y_ch, 2, inplace=inplace),
            DepthConvBlock(y_ch, N),
        )

        self.y_prior_fusion_p = nn.Sequential(
            DepthConvBlock(N * 3 , N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 2, inplace=inplace),
        )
        self.temporal_prior_encoder = ResidualBlockWithStride(N, N)


        self.feature_extractor = FeatureExtractor(N)
        # self.feature_adaptor_p = nn.Conv2d(latent_ch, N, 1)
        # self.lowerbound_scale = LowerBound_sig(0.11)
        # self.dpb=[]
        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(channel_c[:i]), out_dim=channel_c[i] * 2) if i else None
            for i in range(num_slice)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
                EntropyParametersEX(in_dim=N * 2 + channel_c[i] * 2, out_dim=channel_c[i] * 2)
            if i else EntropyParametersEX(in_dim=N * 2, out_dim=channel_c[i] * 2)
            for i in range(num_slice)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=N * 2 + channel_c[i] * 4, out_dim=channel_c[i] * 2)
            if i else EntropyParametersEX(in_dim=N * 2 + channel_c[i] * 2, out_dim=channel_c[i] * 2)
            for i in range(num_slice)
        )
        self.local_context = nn.ModuleList(
            nn.Conv2d(in_channels=channel_c[i], out_channels=channel_c[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(channel_c))
        )
        self.codebook_size = cb_size
        self.quantize = VectorQuantiser(self.codebook_size, y_ch, contras_loss=True)
        self.gaussian_conditional = GaussianConditional(None)

    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    def calculate_bpp(self, likelihood, num_pixels):

        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in likelihood
        )
        return bpp_loss

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)
    def forward(self, x,feature,i_frame_parm,fea_dec=None,ctx_t_f=None):
        # feature = self.apply_feature_adaptor()
        N, _, H, W = x.size()
        num_pixels = N * H * W * 8 * 8 * 4
        ctx, ctx_t = self.feature_extractor(feature)
        if fea_dec is not None:
            ctx_dec,ctx_t = self.feature_extractor(fea_dec)
        else:
            ctx_dec = ctx
        if ctx_t_f is not None:
            _,ctx_t = self.feature_extractor(ctx_t_f)
        y = self.enc_p(x,ctx)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc_p(y_pad)
        z_hat, emb_loss, _ = self.quantize(z)

        num_bits_z = np.log2(self.codebook_size) * z_hat.shape[-2] * z_hat.shape[-1] * z_hat.shape[0]
        z_likelihoods = torch.tensor(num_bits_z, device=z_hat.device)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        q_likelihoods = []
        params = self.hyper_dec_p(z_hat)
        ##########
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        params = params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion_p(
            torch.cat((params, temporal_params,i_frame_parm), dim=1))
        ##########
        params = self.slice_to_y(params, slice_shape)
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor

                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor

                # Non-anchor
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, channel_ctx, params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        q_likelihoods = torch.cat(q_likelihoods, dim=1)

        bpp_y = self.calculate_bpp(y_likelihoods, num_pixels)
        bpp_z = z_likelihoods / num_pixels
        x_hat = self.dec_p(y_hat,ctx_dec)
        bpp = bpp_y + bpp_z
        bits = torch.sum(bpp_y + bpp_z) * num_pixels
        return_dict = {
            "x_hat": x_hat,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            'codebook_loss': emb_loss,
        }
        return return_dict

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scale_enc = ckpt["q_scale_enc"].reshape(-1)
        q_scale_dec = ckpt["q_scale_dec"].reshape(-1)
        return q_scale_enc, q_scale_dec

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)




class LatentCodec(CompressionModel):
    def __init__(self, N=256,latent_ch=4,y_ch=128, cb_size=[8192,2048],ILC_frozen=False):
        super().__init__()
        self.IlatentLC=I_latent_C(N,latent_ch, y_ch, cb_size=cb_size[0])
        self.PlatentLC=P_latent_C(N,latent_ch, y_ch, cb_size=cb_size[1])
        self.feature_extractor = FeatureExtractor(N)
        self.feature_adaptor_p = nn.Conv2d(latent_ch, N, 1)
        self.dpb=[]
        self.ref_parms=None
        if ILC_frozen:
            for param in self.IlatentLC.parameters():
                param.requires_grad = False
    def clear_dpb(self):
        self.dpb.clear()

    def add_ref_frame(self, feature=None, frame=None,):
        ref_frame = RefFrame()
        ref_frame.frame = frame
        ref_frame.feature = feature
        if len(self.dpb) >= 1:
            self.dpb.pop(-1)
        self.dpb.insert(0, ref_frame)

    def apply_feature_adaptor(self):
        return self.feature_adaptor_p(self.dpb[0].feature)

    def forward(self, x,mode='Group1',x0_GT=None,t_fea=None):
        return_dict={}
        if mode=='Group1':
            for i in range(x.size(2)):
                x_slice = torch.squeeze(x[:, :, i:i + 1, :], dim=2)
                if i == 0:
                    f_dict=self.IlatentLC.forward(x_slice)
                    self.ref_parms=f_dict['ref_params'].detach()
                    self.add_ref_frame(f_dict['x_hat'].detach(), None)
                else:
                    feature = self.apply_feature_adaptor()
                    f_dict=self.PlatentLC.forward(x_slice,feature=feature,i_frame_parm=self.ref_parms)
                    self.add_ref_frame(f_dict['x_hat'], None)
                return_dict['frame_{}'.format(i)] = f_dict
            comp_x = []
            for key in return_dict.keys():
                comp_x.append(return_dict[key]['x_hat'].unsqueeze(dim=2))
            comp_x = torch.cat(comp_x, dim=2)
            return_dict['xhat_T'] = comp_x
        else:
            return_dict=self.forward_(x,x0_GT,t_fea)
        return return_dict

    def forward_(self,x,x0_DEC=None,t_fea=None):
        return_dict = {}
        ctx_tf =None
        if x0_DEC!=None and t_fea==None:
            self.ref_parms = self.IlatentLC.enc_i(x0_DEC).detach()
            self.add_ref_frame(x0_DEC.detach(), None)
            feature_dec=self.feature_adaptor_p(x0_DEC.detach())
        elif x0_DEC!=None and t_fea!=None:
            self.ref_parms = t_fea[0]
            ctx_tf=self.feature_adaptor_p(t_fea[1].detach())
            self.add_ref_frame(x[:,:,0,:,:].detach(), None)
            feature_dec=self.feature_adaptor_p(x0_DEC.detach())
        else:
            self.ref_parms = self.IlatentLC.enc_i(x[:,:,0,:,:]).detach()
            self.add_ref_frame(x[:,:,0,:,:].detach(), None)
            feature_dec=None

        for i in range(1,x.size(2)):
            fea_dec= feature_dec if i==1 else None
            x_slice = torch.squeeze(x[:, :, i:i + 1, :], dim=2)
            feature = self.apply_feature_adaptor()
            f_dict = self.PlatentLC.forward(x_slice, feature=feature, i_frame_parm=self.ref_parms,fea_dec=fea_dec,ctx_t_f=ctx_tf)
            self.add_ref_frame(f_dict['x_hat'], None)
            return_dict['frame_{}'.format(i)] = f_dict
        comp_x = []
        for key in return_dict.keys():
            comp_x.append(return_dict[key]['x_hat'].unsqueeze(dim=2))
        comp_x = torch.cat(comp_x, dim=2)
        comp_x = torch.cat([x[:,:,0,:,:].unsqueeze(dim=2),comp_x], dim=2) if x0_DEC==None else torch.cat([x0_DEC.detach().unsqueeze(dim=2),comp_x], dim=2)
        return_dict['xhat_T'] = comp_x
        return return_dict
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
