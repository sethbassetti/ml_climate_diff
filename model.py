from abc import abstractmethod
import math

import torch.nn as nn
import torch.nn.functional as F
import torch

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class WrappedModel:
    """A wrapper around a model that allows for fast sampling using fewer timesteps. Converts from indices in the range
    [0, S] to corresponding original indices from [0, T]"""
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map

        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        """Whenever the model is called, create a map tensor that maps S_i to T_j. Then convert all timesteps
        accordingly before sending t into the model"""
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.query_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Project keys, queries, values and reshape from B x N*C x H x W -> B x N x C x H*W
        queries = self.query_proj(x).reshape(batch_size, self.heads, -1, height * width)
        keys = self.key_proj(x).reshape(batch_size, self.heads, -1, height * width)
        values = self.value_proj(x).reshape(batch_size, self.heads, -1, height * width)

        # Scale down the queries
        queries = queries * self.scale

        # Multiply queries and keys together
        sim = torch.einsum("b h d i, b h d j -> b h i j", queries, keys)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # Multiply attention matrix and values together
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, values)
        out = out.reshape(batch_size, -1, height, width)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Upsample(nn.Module):
    def __init__(self, channels):

        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):

        x = F.interpolate(x, scale_factor=2)
        return self.conv(x)



def get_time_embedding(timesteps, embed_dim):
    """Returns an T x D matrix, where T = number of timesteps and D = embedding dimension """

    assert embed_dim % 2 == 0, "Dimension of timestep embeddings must be divisible by 2"

    device = timesteps.device

    # Half of the indices will be sin and half will be cos
    half_dim = embed_dim // 2

    # Sinusoidal embedding equation
    embedding = math.log(10000) / (half_dim - 1)
    embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)

    # Matrix multiplication to create N x D matrix
    embedding = timesteps[:, None] * embedding[None, :]

    # First half of embeddings are sine and second half are cosine
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    return embedding



class WideResBlock(TimestepBlock):
    """ The main building block of the UNet Architecture. Uses residual connections and sends x through
    two convolutional blocks, along with inserting the time embedding into it"""

    def __init__(self, in_channels, out_channels, time_emb_dim=512, dropout=0.1, use_conv=False, up=False, down=False):
        super().__init__()

        # This variable keeps track of if this resblock upsamples or downsamples
        self.up_down = up or down

        # Define upsampling and downsampling operations for x and h (hidden)
        if up:
            self.x_updsample = Upsample(in_channels)
            self.h_updsample = Upsample(in_channels)
        elif down:
            self.x_updsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
            self.h_updsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        # In block that will process input to residual block
        self.in_block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
        )

        # Transforms the time embedding to our channel dimension
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))

        # Out block process hidden to output
        self.out_block = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')),
        )
        
        # If channels don't change, residual connection can just be added
        if in_channels == out_channels:
            self.res_conv = nn.Identity()

        # Either have a kernel convolution or 1-by convolution
        elif use_conv:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        else:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, time_emb=None):
        """First sends x through a convolutional block. Then injects x with positional information. Then
        sends it through second convolutional block and adds a residual connection"""
        
        # Used to reshape the time embeddings after linear layer
        batch_size = x.shape[0]
        
        # If we are downsampling or upsampling, apply those operations before the convolutional layer
        if self.up_down:
            in_rest, in_conv = self.in_block[:-1], self.in_block[-1]
            h = in_rest(x)
            h = self.h_updsample(h)
            x = self.x_updsample(x)
            h = in_conv(h)
        else:
            h = self.in_block(x)

        # Project time embeddings into current channel dimension and reshape from t= b x c -> b x c x 1 x 1
        time_emb = self.time_mlp(time_emb).reshape(batch_size, -1, 1, 1) if time_emb is not None else 0

        out_norm, out_block = self.out_block[0], self.out_block[1:]

        # Split the time embedding into two dimensions
        scale, shift = torch.chunk(time_emb, 2, dim=1)

        # Apply adaptive group norm to hidden representation and then send through rest of conv
        h = out_norm(h) * (1 + scale) + shift
        h = out_block(h)

        # Return hidden plus the skip connection
        return h + self.res_conv(x)


class UNet(nn.Module):


    def __init__(self, img_start_channels=1, channel_space=64, dim_mults=(1, 2, 2, 2), vartype="fixed",
    blocks_per_res=2, attn_resolutions=(2, 4, 8), dropout=0.1):
        super().__init__()

        self.attn_resolutions = attn_resolutions
        self.blocks_per_res = blocks_per_res
        self.dropout = dropout
        self.channel_space = channel_space
        self.time_dim = channel_space * 4
        self.start_channels = img_start_channels

        # If variance is learned, double output channels so half are mean and half are variance
        self.out_channels = img_start_channels * 2 if vartype == "learned" else img_start_channels
        self.dim_mults = dim_mults

        self.time_linear = nn.Sequential(nn.Linear(self.channel_space, self.time_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.time_dim, self.time_dim))

        res_channels = self.build_input_blocks()
    
        # Defines the channel dimension at the 
        bottleneck_depth = self.dim_mults[-1] * channel_space
        
        # Defines the two bottleneck layers and converts it into a sequential model
        self.bottleneck_1 = WideResBlock(bottleneck_depth, bottleneck_depth, time_emb_dim=self.time_dim, dropout=self.dropout)
        self.mid_attn = Residual(PreNorm(bottleneck_depth, Attention(bottleneck_depth)))
        self.bottleneck_2 = WideResBlock(bottleneck_depth, bottleneck_depth, time_emb_dim=self.time_dim, dropout=self.dropout)

        # Builds the series of expansive blocks leading to the output
        self.expansives = self.build_expansives(res_channels)

        # Output convolutional block that returns image to original channel dim
        self.output_conv = nn.Sequential(nn.GroupNorm(32, self.channel_space),
                                        nn.SiLU(),
                                        zero_module(nn.Conv2d(self.channel_space, self.out_channels, kernel_size=3, padding=1)))
        

    def build_input_blocks(self):
        """ Builds a series of input blocks that downsample spatially but increase channel size"""

        # Initializes the input block, this transforms channel dimension of original image
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential
        (nn.Conv2d(self.start_channels, self.channel_space, 3, padding=1))])

        # Keep track of the channels to be used for residual connections
        res_channels = [self.channel_space]

        # Channels will start at first scale
        ch = self.channel_space * self.dim_mults[0]
        downscale = 1

        # Iterate through each level to construct a conv block and downsample operation
        for level, scale in enumerate(self.dim_mults):
            for _ in range(self.blocks_per_res):
            
                # Define the wide resnet blocks that comprise each layer
                layers = [WideResBlock(ch, scale * self.channel_space, time_emb_dim=self.time_dim, dropout=self.dropout)]
                ch = scale * self.channel_space

                # If we are at an appropriate downscale, add an attention block
                if downscale in self.attn_resolutions:
                    layers.append(Residual(PreNorm(ch, Attention(ch))))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                res_channels.append(ch)

            # If we are not at the last level
            if level != (len(self.dim_mults) - 1):

                out_ch = ch
                # Add a downscale operation
                self.input_blocks.append(TimestepEmbedSequential(WideResBlock(ch, out_ch, time_emb_dim=self.time_dim, down=True, dropout=self.dropout)))

                ch = out_ch
                res_channels.append(ch)
                # Increase the downscale factor by two
                downscale *= 2

        return res_channels

    def build_expansives(self, res_channels):
        """Builds a series of expansive blocks to upsample spatial resolution and downsample channel space"""

        # Initialize a module list to hold all of the expansive blocks
        self.output_blocks = nn.ModuleList([])

        downscale = 2 ** len(self.dim_mults)
        ch = self.channel_space * self.dim_mults[-1]

        for level, scale in list(enumerate(self.dim_mults))[::-1]:
            for i in range(self.blocks_per_res + 1):

                in_res_channels = res_channels.pop()

                layers = [WideResBlock(ch + in_res_channels, self.channel_space * scale, time_emb_dim=self.time_dim, dropout=self.dropout)]
                ch = scale * self.channel_space

                if downscale in self.attn_resolutions:
                    layers.append(Residual(PreNorm(ch, Attention(ch))))
                
                if level and i == self.blocks_per_res:
                    out_ch = ch
                    layers.append(WideResBlock(ch, out_ch, time_emb_dim=self.time_dim, up=True, dropout=self.dropout))
                    downscale //= 2

                # Create a block that holds the convolutional block and the upsampling operation
                self.output_blocks.append(TimestepEmbedSequential(*layers))



    def forward(self, x, t):
        
        # This is the list of residuals to pass across the UNet
        h = [] 

        # Create a fixed time embedding from t and project it to a higher dimensional space
        t = get_time_embedding(t, self.channel_space)
        t = self.time_linear(t)


        #x = self.input_conv(x)

        # Iterate through each level of the contrastive portion of the model
        for block in self.input_blocks:  
            x = block(x, t)
            h.append(x)

        # Send x through bottleneck at bottom of model
        x = self.bottleneck_1(x, t)
        x = self.mid_attn(x)
        x = self.bottleneck_2(x, t)

        # Iterate through the expansive, upsampling part of the model
        for block in self.output_blocks:

            # Send x through one expansive level of model
            x = torch.cat((x, h.pop()), dim=1)      # Concatenate residual connection to start of x
            x = block(x, t)                         # Send x through conv block
            
            

        # Send x through output block that squashes channel space and return output
        out = self.output_conv(x)
        return out


if __name__ == "__main__":
    img = torch.randn(4, 3, 32, 32)
    timesteps = torch.randint(0, 1000, (4,))

    model = UNet(img_start_channels=3, channel_space=128)
    out = model(img, timesteps)
    print(out.shape)