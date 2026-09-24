import torch.nn as nn
import torch
from einops import rearrange
from compressai.layers import conv3x3
import torch.nn.functional as F


def subpel_conv1x1(in_ch, out_ch, r=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)



class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out



class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2, inplace=False):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out

class Encoder_I(nn.Module):
    def __init__(self, in_nc, M):
        super().__init__()

        self.g_a = nn.Sequential(
            ResidualBlock(in_nc, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlockWithStride(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            conv3x3(M, M)
        )

    def forward(self, x):
        return self.g_a(x)


class Decoder_I(nn.Module):
    def __init__(self, M):
        super().__init__()

        self.g_s = nn.Sequential(
            conv3x3(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
        )

    def forward(self, x):
        return self.g_s(x)




class HyperEncoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()

        self.hyper_enc = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N),
            ResidualBlockWithStride(N, N),
        )

    def forward(self, x):
        return self.hyper_enc(x)


class HyperDecoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(N, M),
            ResidualBlockUpsample(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M,M),
        )

    def forward(self, x):
        return self.hyper_dec(x)


class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(224, 128, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, channel_params):
        return self.fushion(channel_params)


class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            nn.GELU(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            nn.GELU(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
        )

    def forward(self, params):
        return self.fusion(params)



class FeaturePool():
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if features.size(
                    0) > self.pool_size:  # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features



class VectorQuantiser(nn.Module):

    def __init__(self, num_embed, embed_dim, beta=0.25, distance='l2',
                 anchor='closest', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))
        self.register_buffer('usage', torch.zeros(self.num_embed, dtype=torch.int), persistent=False)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
            torch.sum(self.embedding.weight ** 2, dim=1) + \
            2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)

        loss = 0.0
        perplexity = None

        if self.training:
            # compute loss for embedding
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
            # preserve gradients, STE
            z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        b, _, h, w = z_q.shape
        encoding_indices = rearrange(encoding_indices, '(b h w) -> b h w', b=b, h=h, w=w).contiguous()

        min_encodings = encodings

        if not self.training:
            for idx in range(self.num_embed):
                self.usage[idx] += (encoding_indices == idx).sum()

        # online clustered reinitialisation for unoptimized points
        if self.training:
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob * self.num_embed * 10) / (1 - self.decay) - 1e-3).unsqueeze(
                    1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True

            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_embed)):, :].mean(dim=0,
                                                                                                       keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss += contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)