# external imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# autoencoder modules

class Encoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(Encoder, self).__init__()
        self.add_module('conv', nn.Conv2d(f_in, f_out, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        skip = x
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        return F.max_pool2d(x, kernel_size=2), skip

class Decoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(Decoder, self).__init__()
        self.add_module('conv1', nn.ConvTranspose2d(f_in, f_out, kernel_size=3, stride=1, padding=1))
        self.add_module('conv2', nn.ConvTranspose2d(f_out, f_out, kernel_size=3, stride=1, padding=1))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.size()[-2:])
        x = torch.cat((x, skip), dim=-3)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_channels, output_channels=3, stages=5, hidden=48):
        super(Autoencoder, self).__init__()
        stages = max(2, stages) # ensure min 2 stages
        # input layer
        self.add_module("input", nn.Conv2d(input_channels, hidden, kernel_size=3, stride=1, padding=1))
        # encoder modules
        self.encoders = nn.ModuleList()
        for i in range(stages):
            self.encoders.append(Encoder(hidden, hidden))
        # latent space
        self.add_module("latent", nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1))
        # decoder modules
        self.decoders = nn.ModuleList()
        for i in range(stages):
            if i == 0: # first decoder: hidden + skip_connections feature maps
                self.decoders.append(Decoder(2*hidden, 2*hidden))
            else: # intermediate decoder: 2*hidden + skip_connections feature maps
                self.decoders.append(Decoder(3*hidden, 2*hidden))
        # output layer
        self.add_module('dropout', nn.Dropout2d(0.1))
        self.add_module("output", nn.ConvTranspose2d(2*hidden, output_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.input(x * 2 - 1) # zero centered input
        # encoder stages
        skip_connections = []
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x)
            skip_connections.append(skip)
        # latent space
        x = self.latent(x)
        # decoder stages
        skip_connections.reverse()
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        x = self.dropout(x)
        return self.output(x) * 0.5 + 0.5 # output in [0, 1]

class AutoencoderDualF24(nn.Module):
    def __init__(self):
        super(AutoencoderDualF24, self).__init__()
        self.add_module("aec_direct", Autoencoder(15, 3, stages=5, hidden=48))
        self.add_module("aec_indirect", Autoencoder(15, 3, stages=5, hidden=48))
        self.add_module("output", nn.ConvTranspose2d(6, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        direct = self.aec_direct(x[..., 0:15, :, :])
        indirect = self.aec_indirect(torch.cat((x[..., 0:3, :, :], direct, x[..., 15:24, :, :]), dim=-3))
        x = self.output(torch.cat((direct * 2 - 1, indirect * 2 - 1), dim=-3)) * 0.5 + 0.5
        return x

class AutoencoderDualF24Big(nn.Module):
    def __init__(self):
        super(AutoencoderDualF24Big, self).__init__()
        self.add_module("aec_direct", Autoencoder(15, 12, stages=5, hidden=48))
        self.add_module("aec_indirect", Autoencoder(24, 12, stages=5, hidden=48))
        self.add_module("output", nn.ConvTranspose2d(24, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        direct = self.aec_direct(x[..., 0:15, :, :])
        indirect = self.aec_indirect(torch.cat((x[..., 0:3, :, :], direct, x[..., 15:24, :, :]), dim=-3))
        x = self.output(torch.cat((direct * 2 - 1, indirect * 2 - 1), dim=-3)) * 0.5 + 0.5
        return x

# -----------------------------------------------------------
# temporal module adapters

class TemporalAdapter(nn.Module):
    def __init__(self, model):
        super(TemporalAdapter, self).__init__()
        self.add_module("model", model)

    def forward(self, x):
        assert x.dim() > 4, "Temporal autoencoder needs 5D data!"
        frames = []
        for i in range(x.size(1)):
            frames.append(self.model(x[:, i]))
        return torch.stack(frames, dim=1)

# -----------------------------------------------------------
# reprojection module adapters

class ReprojSelectAdapter(nn.Module):
    def __init__(self, autoencoder, input_channels, n_frames):
        super(ReprojSelectAdapter, self).__init__()
        self.add_module("autoencoder", autoencoder)
        self.add_module("selector", Autoencoder((2*n_frames+1)*input_channels, input_channels, stages=5, hidden=48))

    def forward(self, x):
        assert x.dim() > 4, "Reprojection autoencoder needs 5D data!"
        # run feature selector
        x = self.selector(x.reshape(x.size(0), -1, x.size(-2), x.size(-1)))
        # run denoiser on selected features
        return self.autoencoder(x)

class ReprojFeatureSelectAdapter(nn.Module):
    def __init__(self, autoencoder, input_channels, n_frames):
        super(ReprojFeatureSelectAdapter, self).__init__()
        self.add_module("autoencoder", autoencoder)
        assert input_channels % 3 == 0, "Input channel size mismatch (% 3 != 0)!"
        self.selectors = nn.ModuleList()
        for i in range(input_channels//3):
            self.selectors.append(Autoencoder((2*n_frames+1)*3, stages=2, hidden=16))

    def forward(self, x):
        assert x.dim() > 4, "Reprojection autoencoder needs 5D data!"
        # run feature selectors on feature maps
        selected_features = []
        for i, selector in enumerate(self.selectors):
            f = x[:, :, i*3:(i+1)*3].reshape(x.size(0), -1, x.size(-2), x.size(-1))
            selected_features.append(selector(f))
        # concat and run main denoiser
        return self.autoencoder(torch.cat(selected_features, dim=-3))

# -----------------------------------------------------------
# discriminator modules

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(f_in, f_out, drop=0.25, bn=True):
            block = [nn.Conv2d(f_in, f_out, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(inplace=True, negative_slope=0.2),
                    nn.Dropout2d(drop)]
            if bn: block.append(nn.BatchNorm2d(f_out))
            return block

        self.model = nn.Sequential(
                *discriminator_block(3, 32, bn=False),
                *discriminator_block(32, 64, bn=True),
                *discriminator_block(64, 128, bn=True),
                *discriminator_block(128, 256, bn=True),
                *discriminator_block(256, 512, bn=True))
        self.pool = nn.AvgPool2d(4)

    def forward(self, x):
        x = x * 2 - 1 # zero centered input
        x = self.model(x)
        x = self.pool(x)
        return x # output in [-1, 1]

class Discriminator256x3(nn.Module):
    # from https://arxiv.org/pdf/1807.00734.pdf DCGAN 256x256, it's huge
    def __init__(self):
        super(Discriminator256x3, self).__init__()
        self.add_module('conv1', nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('conv3', nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('norm3', nn.BatchNorm2d(128))
        self.add_module('conv4', nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('norm4', nn.BatchNorm2d(256))
        self.add_module('conv5', nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('norm5', nn.BatchNorm2d(512))
        self.add_module('conv6', nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('norm6', nn.BatchNorm2d(1024))
        self.add_module('conv7', nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=1, bias=False))

    def forward(self, x):
        x = x * 2 - 1 # zero centered input
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = self.norm3(F.leaky_relu(self.conv3(x), negative_slope=0.2))
        x = self.norm4(F.leaky_relu(self.conv4(x), negative_slope=0.2))
        x = self.norm5(F.leaky_relu(self.conv5(x), negative_slope=0.2))
        x = self.norm6(F.leaky_relu(self.conv6(x), negative_slope=0.2))
        return self.conv7(x) # output in [-1, 1]
