import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.half_dim = dim // 2
        self.max_period = max_period

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        emb = math.log(self.max_period) / (self.half_dim - 1)
        emb = torch.exp(
            torch.arange(self.half_dim, device=t.device, dtype=t.dtype) * -emb
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int, time_cond: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width)
        if time_cond:
            self.linear_2 = nn.Linear(2 * width, width)
        else:
            self.linear_2 = nn.Linear(width, width)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width)
        self.time_cond = time_cond

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        if self.time_cond:
            # repeat time embedding for seq_len
            # [Batch_Size, Seq_Len, Width]
            time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    """

    def __init__(
        self,
        input_dim,
        embed_dim=256,
        scale=10,
    ):
        super(GaussianFourierFeatureTransform, self).__init__()
        self.b = torch.randn(input_dim, embed_dim) * scale
        self.pi = 3.14159265359

    def forward(self, v: torch.FloatTensor) -> torch.FloatTensor:
        x_proj = torch.matmul(2 * self.pi * v, self.b.to(v.device).to(v.dtype))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, dim_cond: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid(),
        )
        self.to_beta = nn.Linear(dim_cond, dim, bias=False)

    def _norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self, x: torch.FloatTensor, cond: torch.FloatTensor
    ) -> torch.FloatTensor:
        output = self._norm(x)
        if cond.ndim == 2:
            cond = rearrange(cond, "b d -> b 1 d")
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return output * gamma + beta


class AdaptiveLayerscale(nn.Module):
    def __init__(
        self, dim: int, dim_cond: int, adaln_zero_bias_init_value: float = -2.0
    ):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(
        self, x: torch.FloatTensor, cond: torch.FloatTensor
    ) -> torch.FloatTensor:
        if cond.ndim == 2:
            cond = rearrange(cond, "b d -> b 1 d")
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma.sigmoid()

"""
    VAE modules
"""

from typing import List, Callable, Union, Any, TypeVar, Tuple
from abc import abstractmethod
Tensor = TypeVar('torch.tensor')

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor, return_indices=False) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        if return_indices:
            return quantized_latents.permute(0, 3, 1, 2).contiguous(), \
                    vq_loss, \
                    encoding_inds.view(latents.shape[0], latents.shape[1], latents.shape[2])

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 128, 256, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def get_depth_tokens(gt_img_tensor, decoded_tokens, checkpoint_path, temp=0.2):
    assert checkpoint_path is not None, "Checkpoint path is None"
    # gt_img_tensor: [B, 3, 256, 256]
    gt_img_tensor = F.interpolate(gt_img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    gt_img_tensor = gt_img_tensor.float().div(255.0).cuda() 

    # load VQVAE model
    vqvae_model = VQVAE(in_channels=3,embedding_dim=160,num_embeddings=256,img_size=256)
    raw_ckpt = torch.load(checkpoint_path, map_location='cuda')

    state_dict = raw_ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.replace("model.", "", 1)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    vqvae_model.load_state_dict(new_state_dict)
    vqvae_model.eval().cuda()

    with torch.no_grad():
        z_e = vqvae_model.encode(gt_img_tensor)[0]  # [B, D, 16, 16]
        z_q, _, indices = vqvae_model.vq_layer(z_e, return_indices=True)  # indices: [B, 16, 16]

        codebook = vqvae_model.vq_layer.embedding.weight  # [K=256, D=160]
        # codebook = F.normalize(codebook, dim=-1)

    # reshape decoded
    B, T, D = decoded_tokens.shape 
    # logits = torch.matmul(decoded_tokens.view(B*T, D), codebook.t()) / temp # [B*T, 256]
    decoded_tokens = decoded_tokens.view(B * T, D)
    codebook = codebook.t()

    x_norm_sq = (decoded_tokens ** 2).sum(dim=1, keepdim=True)
    c_norm_sq = (codebook ** 2).sum(dim=0, keepdim=True)
    cross_term = decoded_tokens @ codebook

    dists = x_norm_sq + c_norm_sq - 2 * cross_term
    logits = -dists/temp  # [B*T, 256]
        
    return indices.view(z_q.shape[0], -1), logits 

class DepthDecoder(nn.Module):
    """
    depth_decoder simple version
    """
    def __init__(self, input_dim=1024, output_dim=160):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [B, 256, 1024]
        return self.proj(x)  # output: [B, 256, 160]

# a more complex decoder    
class ConvDepthDecoder(nn.Module):
    def __init__(self, input_channels=1024, output_dim=160):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, output_dim, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, 256, 1024] → reshape to [B, 1024, 16, 16]
        B, T, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, 16, 16)
        x = self.decoder(x)  # [B, 160, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 256, 160]
        return x