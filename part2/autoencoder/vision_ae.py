import torch
import numpy as np

class VisualEncoder(torch.nn.Module):
    def __init__(self,
        visual_embedding_dim, img_dim, grayscale,
        kernel_sizes: list[tuple[int]] = [(4,5)],
        kernel_strides: list[int] = [3],
        channels: list[int] = [8],
    ):
        super().__init__()

        inc = 1 if grayscale else 3
        img_dim_out = img_dim

        encoder_conv_layers = []
        for i in range(len(kernel_sizes)):
            ksize = kernel_sizes[i]
            stride = kernel_strides[i]
            encoder_conv_layers.append(torch.nn.Conv2d(
                in_channels = (inc if i == 0 else channels[i-1]),
                out_channels = channels[i],
                kernel_size = ksize,
                stride = stride,
            ))
            encoder_conv_layers.append(torch.nn.BatchNorm2d(channels[i], track_running_stats=False))

            img_dim_out = [
                int((img_dim_out[i] - ksize[i])/stride + 1)
                for i in range(len(img_dim))
            ]
            
            encoder_conv_layers.append(torch.nn.ReLU())
            # encoder_conv_layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            
        self.encoder_conv = torch.nn.Sequential(*encoder_conv_layers)
        
        self.img_dim_out = img_dim_out
        
        print(f'Encoder convolution layer: {self.encoder_conv}')
        print(f'Final convolution size: {channels[-1]}x{img_dim_out}')
        print(f'Flattens to {channels[-1]*np.prod(img_dim_out)}\n')
            
        self.flatten = torch.nn.Flatten(start_dim=1)

        self.encoder_lin = torch.nn.Sequential(
            torch.nn.Linear(channels[-1]*np.prod(img_dim_out), visual_embedding_dim),
            torch.nn.Sigmoid()
        )
        print(f'Encoder linear layer: {self.encoder_lin}')

    def forward(self, img):
        
        img_conv = self.encoder_conv(img)
        img_conv_flatten = self.flatten(img_conv)

        embeddings = self.encoder_lin(img_conv_flatten)

        return embeddings

class VisualDecoder(torch.nn.Module):
    def __init__(self,
        visual_embedding_dim, img_dim_out, grayscale,
        kernel_sizes: list[tuple[int]] = [(4,5)],
        kernel_strides: list[int] = [3],
        channels: list[int] = [8],
    ):
        super().__init__()

        inc = 1 if grayscale else 3

        self.decoder_lin = torch.nn.Linear(visual_embedding_dim, channels[-1]*np.prod(img_dim_out))
        print(f'\nDecoder linear layer: {self.decoder_lin}\n')

        self.unflatten = torch.nn.Unflatten(
            dim=1,
            unflattened_size=channels[-1:]+img_dim_out
        )
            
        decoder_conv_layers = []
        for i in range(len(kernel_sizes)-1, -1, -1):
            decoder_conv_layers.append(torch.nn.ConvTranspose2d(
                in_channels = channels[i],
                out_channels = (inc if i == 0 else channels[i-1]),
                kernel_size = kernel_sizes[i],
                stride = kernel_strides[i],
            ))
            decoder_conv_layers.append(torch.nn.BatchNorm2d(inc if i == 0 else channels[i-1], track_running_stats=False))

            decoder_conv_layers.append(
                torch.nn.Sigmoid() if i == 0 else torch.nn.ReLU()
            )

        self.decoder_conv = torch.nn.Sequential(*decoder_conv_layers)
        print(f'Decoder convolution layer: {self.decoder_conv}')

    def forward(self, embeddings):
        img_conv = self.decoder_lin(embeddings)

        img_conv = self.unflatten(img_conv)
        
        img_reconstructed = self.decoder_conv(img_conv)
        
        return img_reconstructed
