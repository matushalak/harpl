import torch
from harpl.networks.networks import EncoderNetwork, DecoderNetwork
import torch.nn as nn
import numpy as np


class MLPEncoder(nn.Module):
    """MLP Encoder network (implemented as a convolutional network for 1D data)
    
    Args:
        input_dim (int): Dimension of the input. Default: 14
        output_dim (int): Dimension of the encoded representation. Default: 256
        use_batch_norm (bool): Whether to use batch normalization. Default: False
        n_layers (int): Number of hidden layers. Default: 3

    Attributes:
        layers (nn.Sequential): MLP network
    
    Note:
        This is a simple MLP encoder network with three hidden layers.
        We implement the MLP as a convolutional network with stride 1 over the sequence length for efficient computation
        on small sequence lengths. This requires the inputs to have channels as the second dimension, hence why forward 
        expects an input tensor of shape (B, C, L) instead of (B, L, C) as for other datasets.
        The output of the network is still of shape (B, L, output_dim) as usual.
    """

    def __init__(self, input_dim=14, output_dim=256, use_batch_norm=False, n_layers=3):
        super().__init__()

        norm_layer = nn.BatchNorm1d
        self.layers = nn.Sequential()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else output_dim
            self.layers.add_module(
                f"dense{i}",
                nn.Conv1d(in_channels, output_dim, kernel_size=1, stride=1, padding=0),
            )
            if use_batch_norm:
                self.layers.add_module(f"batchNorm{i}", norm_layer(output_dim))
            self.layers.add_module(f"relu{i}", nn.ReLU())
    
    def forward(self, x):
        # return contiguous tensor
        if len(x.shape) == 3:
            # (B, Cin, L) -> (B, L, Cout)
            x = self.layers(x).transpose(1, 2).contiguous()
        elif len(x.shape) == 5:
            # (B, L, C, H, W) -> (B, L, C*H*W) -> (B, C*H*W, L) -> (B, L, Cout)
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self.layers(x).transpose(1, 2).contiguous()
        return x.contiguous()


# Added for HARPL
class MLPDecoder(nn.Module):
    """MLP Decoder network (implemented as a convolutional network for 1D data)

    Args:
        input_dim (int): Dimension of the input. Default: 256
        output_dim (int): Dimension of the decoded representation. Default: 14
        use_batch_norm (bool): Whether to use batch normalization. Default: False
        n_layers (int): Number of hidden layers. Default: 3

    Attributes:
        layers (nn.Sequential): MLP network
    
    Note:
        This is a simple MLP decoder network with three hidden layers.
        We implement the MLP as a convolutional network with stride 1 over the sequence length for efficient computation
        on small sequence lengths. This requires the inputs to have channels as the second dimension, hence why forward 
        expects an input tensor of shape (B, C, L) instead of (B, L, C) as for other datasets.
        The output of the network is still of shape (B, L, output_dim) as usual.
    """

    def __init__(self, input_dim=256, output_dim=14, use_batch_norm=False, n_layers=3):
        super().__init__()

        norm_layer = nn.BatchNorm1d
        self.layers = nn.Sequential()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else output_dim
            self.layers.add_module(
                f"dense{i}",
                nn.Conv1d(in_channels, output_dim, kernel_size=1, stride=1, padding=0),
            )
            if use_batch_norm:
                self.layers.add_module(f"batchNorm{i}", norm_layer(output_dim))
            self.layers.add_module(f"relu{i}", nn.ReLU())
    
    def forward(self, x):
        # return contiguous tensor
        if len(x.shape) == 3:
            # (B, Cin, L) -> (B, L, Cout)
            x = self.layers(x).transpose(1, 2).contiguous()
        elif len(x.shape) == 5:
            # (B, L, C, H, W) -> (B, L, C*H*W) -> (B, C*H*W, L) -> (B, L, Cout)
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self.layers(x).transpose(1, 2).contiguous()
        return x.contiguous()


class Conv2dEncoder(EncoderNetwork):
    """Conv2d Encoder network (implemented as a convolutional network for 2D data, where width is the sequence length)
    
    Args:
        n_in_channels (int): Number of input channels. Default: 1
        channel_dim (int): Dimension of the encoded representation. Default: 512
        use_batch_norm (bool): Whether to use batch normalization. Default: False
        n_layers (int): Number of convolutional layers. Default: 1
        kernel_size (list): List of kernel sizes. Default: [(5,5)]
        stride (list): List of strides. Default: [(2,2)]
        padding (list): List of paddings. Default: [(2,2)]
        max_pool_size (list): Encoder max-pool sizes to invert with upsampling. Default: [(2,2)]
        return_full_feature_map (bool): Kept for API compatibility. Default: False

    Attributes:
        backbone (nn.Sequential): Transposed-convolutional network
        decoded_dim (int): Dimension of the decoded representation

    Note:
        This mirrors Conv2dEncoder in reverse order. Encoder max-pooling is inverted
        with nearest-neighbor upsampling because exact MaxUnpool2d would require
        pooling indices from the encoder forward pass.
    """
    def __init__(self, n_in_channels=1, channel_dim=512, use_batch_norm=False, n_layers=1, kernel_size=[(5,5)], stride=[(2,2)], padding=[(2,2)], max_pool_size=[(2,2)], return_full_feature_map=False):
        super(Conv2dEncoder, self).__init__(encoded_dim=channel_dim)

        # if kernel_size, stride, padding, and max_pool_size are only given for one layer, repeat them for n_layers
        if len(kernel_size) == 1 and n_layers > 1:
            kernel_size = kernel_size * n_layers
            stride = stride * n_layers
            if padding is not None:
                padding = padding * n_layers
            if max_pool_size is not None:
                max_pool_size = max_pool_size * n_layers

        for i in range(n_layers):
            in_channels = n_in_channels if i == 0 else channel_dim
            self.backbone.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channel_dim,
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=padding[i] if padding is not None else 0
                )
            )
            if use_batch_norm:
                self.backbone.add_module(f"batchNorm{i}", nn.BatchNorm2d(channel_dim))
            self.backbone.add_module(f"relu{i}", nn.ReLU())
            if max_pool_size is not None:
                self.backbone.add_module(f"pool{i}", nn.MaxPool2d(max_pool_size[i]))

        self.return_full_feature_map = return_full_feature_map

        self.kernel_size = np.array(kernel_size)[:, 0]
        self.stride = np.array(stride)[:, 0]
        self.padding = np.array(padding)[:, 0] if padding is not None else np.zeros(n_layers)
        self.max_pool_size = np.array(max_pool_size)[:, 0] if max_pool_size is not None else np.ones(n_layers)
        self.channel_dim = channel_dim

    def forward(self, x):
        # audio data is (B, H, L)
        if len(x.shape) == 3:
            # (B, H, L) -> (B, 1, H, L) -> (B, C, H, L) -> (B, C, L) 
            z = super().forward(x.unsqueeze(1)).mean(-2)
            # (B, C, L) -> (B, L, C)
            return z.transpose(1, 2).contiguous()
        # XXX visual stimulus is (B, L, C, H, W)
        elif len(x.shape) > 3:
            is_video = len(x.shape) == 5
            if is_video:
                L = x.size(1)
                # (B, L, C, H, W) -> (B*L, C, H, W)
                x = x.reshape(-1, *x.shape[2:])
            z = super().forward(x)
            if is_video:
                # (B*L, C, H, W) -> (B, L, C, H, W)
                z = z.reshape(-1, L, z.size(1), z.size(2), z.size(3))

            if self.return_full_feature_map:
                return z
            # (B, L, C, H, W) -> (B, L, C) for video and (B, Nx*Ny, C) for image
            return z.mean(-1).mean(-1)

    def get_output_spatial_shape(self, input_shape):
        shape = input_shape if isinstance(input_shape, int) else input_shape[0]
        for i in range(len(self.kernel_size)):
            shape = (shape - self.kernel_size[i] + 2 * self.padding[i]) // self.stride[i] + 1
            if self.max_pool_size is not None:
                shape = (shape - self.max_pool_size[i]) // self.max_pool_size[i] + 1
        return int(shape)


class Conv2dDecoder(DecoderNetwork):
    """Conv2d Decoder network (implemented as a transposed-convolutional network for 2D data, where width is the sequence length)
    
    Args:
        n_in_channels (int): Number of input channels. Default: 512
        channel_dim (int): Dimension of the decoded representation. Default: 1
        use_batch_norm (bool): Whether to use batch normalization. Default: False
        n_layers (int): Number of convolutional layers. Default: 1
        kernel_size (list): List of kernel sizes. Default: [(5,5)]
        stride (list): List of strides. Default: [(2,2)]
        padding (list): List of paddings. Default: [(2,2)]
        max_pool_size (list): List of max pool sizes. Default: [(2,2)]
        return_full_feature_map (bool): Whether to return the full feature maps of the last layer (only used for vision). Default: False

    Attributes:
        backbone (nn.Sequential): Convolutional network
        encoded_dim (int): Dimension of the encoded representation

    Note:
        This is a tweakable Conv2d encoder network with one or more hidden layers.
        The network consists of convolutional layers with optional batch normalization and ReLU activation.
        The input is expected to be of shape (B, H, L) for audio where B is the batch size, H is the height, and L is the sequence length,
        and (B, L, C, H, W) for video where C, H and W are number of the channels, the height and width of each frame respectively.
        If `return_full_features` is set to True, the output of the network will be the full feature maps of the last layer, i.e. the output 
        of the network is of shape (B, L, C, H, W). Otherwise, the output is obtained by averaging over height for audio and over width and 
        height for video, resulting in a tensor of shape (B, L, C).
    """
    def __init__(self, n_in_channels=512, channel_dim=1, use_batch_norm=False, n_layers=1, 
                 kernel_size=[(5,5)], stride=[(2,2)], padding=[(2,2)], max_pool_size=[(2,2)], 
                 return_full_feature_map=False):
        super(Conv2dDecoder, self).__init__(decoded_dim=channel_dim)

        # if kernel_size, stride, padding, and max_pool_size are only given for one layer, repeat them for n_layers
        if len(kernel_size) == 1 and n_layers > 1:
            kernel_size = kernel_size * n_layers
            stride = stride * n_layers
            if padding is not None:
                padding = padding * n_layers
            if max_pool_size is not None:
                max_pool_size = max_pool_size * n_layers

        for i, enc_i in enumerate(range(n_layers - 1, -1, -1)):
            in_channels = n_in_channels
            out_channels = channel_dim if i == n_layers - 1 else n_in_channels
            if max_pool_size is not None:
                self.backbone.add_module(
                    f"unpool{enc_i}",
                    nn.Upsample(scale_factor=max_pool_size[enc_i], mode="nearest")
                )
            self.backbone.add_module(
                f"convTranspose{enc_i}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size[enc_i],
                    stride=stride[enc_i],
                    padding=padding[enc_i] if padding is not None else 0,
                    output_padding=(
                        (stride[enc_i][0] - 1, stride[enc_i][1] - 1)
                        if stride[enc_i][0] > 1 or stride[enc_i][1] > 1 else 0
                    )
                )
            )
            if i != n_layers - 1:
                if use_batch_norm:
                    self.backbone.add_module(f"batchNorm{enc_i}", nn.BatchNorm2d(out_channels))
                self.backbone.add_module(f"relu{enc_i}", nn.ReLU())

        self.return_full_feature_map = return_full_feature_map

        self.kernel_size = np.array(kernel_size)[:, 0]
        self.stride = np.array(stride)[:, 0]
        self.padding = np.array(padding)[:, 0] if padding is not None else np.zeros(n_layers)
        self.max_pool_size = np.array(max_pool_size)[:, 0] if max_pool_size is not None else np.ones(n_layers)
        self.channel_dim = channel_dim

    def forward(self, z):
        """"
        NOTE: here x and z names are be swapped since we are going from latent (z) to input space (x)
        """
        # encoded audio sequence is (B, L, C)
        if len(z.shape) == 3:
            # (B, L, C) -> (B, C, 1, L) -> (B, Cout, H, Lout)
            x = super().forward(z.transpose(1, 2).unsqueeze(-2))
            if x.size(1) == 1:
                return x.squeeze(1).contiguous()
            return x.contiguous()
        # XXX encoded visual feature map is (B, L, C, H, W) for video or (B, C, H, W) for image
        elif len(z.shape) > 3:
            is_video = len(z.shape) == 5
            if is_video:
                L = z.size(1)
                # (B, L, C, H, W) -> (B*L, C, H, W)
                z = z.reshape(-1, *z.shape[2:])
            x = super().forward(z)
            if is_video:
                # (B*L, C, H, W) -> (B, L, C, H, W)
                x = x.reshape(-1, L, x.size(1), x.size(2), x.size(3))
            return x.contiguous()

    def get_output_spatial_shape(self, input_shape):
        shape = input_shape if isinstance(input_shape, int) else input_shape[0]
        for i in range(len(self.kernel_size) - 1, -1, -1):
            if self.max_pool_size is not None:
                shape = shape * self.max_pool_size[i]
            shape = (shape - 1) * self.stride[i] - 2 * self.padding[i] + self.kernel_size[i]
            if self.stride[i] > 1:
                shape += self.stride[i] - 1
        return int(shape)
