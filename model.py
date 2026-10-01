import torch
from torch import nn
import torch.nn.functional as F


def nmse(pred,y):
    return (pred-y).square().sum((1,2,3))/y.square().sum((1,2,3)).clamp_min(1e-12)


class Block(nn.Module):
    def __init__(self,inc,out):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(inc,out,3,padding=1),nn.GroupNorm(8,out),nn.SiLU(),
                               nn.Conv2d(out,out,3,padding=1),nn.GroupNorm(8,out),nn.SiLU())
    def forward(self,x):return self.net(x)


class NoiseFiLM(nn.Module):
    """Channel-wise feature modulation conditioned on one scalar noise statistic per scene.

    The final linear layer is zero initialized, so a freshly initialized FiLM block starts as
    an exact identity: F' = F * (1 + 0) + 0. This makes adding the conditioning path low risk.
    """
    def __init__(self,channels,hidden=32):
        super().__init__()
        self.channels=int(channels)
        self.net=nn.Sequential(
            nn.Linear(1,hidden),
            nn.SiLU(),
            nn.Linear(hidden,2*self.channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,noise_scalar):
        if noise_scalar.ndim==1:
            noise_scalar=noise_scalar[:,None]
        if noise_scalar.ndim!=2 or noise_scalar.shape[1]!=1:
            raise ValueError('noise_scalar must have shape [B] or [B,1]')
        gamma,beta=self.net(noise_scalar).chunk(2,dim=1)
        return gamma[:,:,None,None],beta[:,:,None,None]


def apply_film(features,film,noise_scalar):
    gamma,beta=film(noise_scalar)
    return features*(1.+gamma)+beta


class Reconstructor(nn.Module):
    def __init__(self,bridge,hybrid=True,width=32,noise_conditioning='film'):
        super().__init__()
        self.bridge=bridge;self.hybrid=hybrid
        self.noise_conditioning=noise_conditioning
        if noise_conditioning not in ('film','none'):
            raise ValueError("noise_conditioning must be 'film' or 'none'")
        self.enc1=Block(37,width);self.enc2=Block(width,width*2);self.middle=Block(width*2,width*4)
        self.dec2=Block(width*6,width*2);self.dec1=Block(width*3,width)
        if noise_conditioning=='film':
            hidden=max(16,width)
            self.noise_film_1=NoiseFiLM(width,hidden)
            self.noise_film_2=NoiseFiLM(width*2,hidden)
            self.noise_film_3=NoiseFiLM(width*4,hidden)
        self.out=nn.Conv2d(width,4,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    @staticmethod
    def normalized_log_noise(variance,scale):
        # The model input is divided by ``scale``. Condition on the corresponding normalized
        # noise variance, averaged across the eight deformation views.
        value=torch.log10((variance.mean(1)/scale.square()).clamp_min(1e-12))
        # Very broad clipping prevents pathological values while leaving the practical range
        # untouched. Scaling keeps the MLP input in a comfortable numerical range.
        return value.clamp(-12.,6.)/6.

    def forward(self,x,variance,scale):
        base=self.bridge(x,variance) if self.hybrid else torch.zeros_like(x[:,:4])
        raw_log_noise=torch.log10((variance.mean(1)/scale.square()).clamp_min(1e-12))
        noise=raw_log_noise[:,None,None,None].expand(-1,1,25,25)
        conditioning=self.normalized_log_noise(variance,scale)

        a=self.enc1(torch.cat((x,base,noise),1))
        if self.noise_conditioning=='film':
            a=apply_film(a,self.noise_film_1,conditioning)

        b=self.enc2(F.avg_pool2d(a,2))
        if self.noise_conditioning=='film':
            b=apply_film(b,self.noise_film_2,conditioning)

        c=self.middle(F.avg_pool2d(b,2))
        if self.noise_conditioning=='film':
            c=apply_film(c,self.noise_film_3,conditioning)

        d=self.dec2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e=self.dec1(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return base+self.out(e)
