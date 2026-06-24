import torch
import torch.nn.functional as F

from pnpflow.utils import (
    create_downsampling_matrix,
    downsample,
    gaussian_2d_kernel,
    random_mask,
    square_mask,
    upsample,
)


class Degradation:
    def H(self, x):
        raise NotImplementedError()

    def H_adj(self, x):
        raise NotImplementedError()


class Denoising(Degradation):
    def H(self, x):
        return x

    def H_adj(self, x):
        return x


class BoxInpainting(Degradation):
    def __init__(self, half_size_mask):
        super().__init__()
        self.half_size_mask = half_size_mask

    def H(self, x):
        return square_mask(x, self.half_size_mask)

    def H_adj(self, x):
        return square_mask(x, self.half_size_mask)


class RandomInpainting(Degradation):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def H(self, x):
        return random_mask(x, self.p)

    def H_adj(self, x):
        return random_mask(x, self.p)


class GaussianDeblurring(Degradation):
    def __init__(self, sigma_blur, kernel_size, mode="fft",
                 num_channels=3, dim_image=128, device="cuda"):
        super().__init__()
        self.mode = mode
        self.sigma = sigma_blur
        self.kernel_size = kernel_size
        self.kernel = gaussian_2d_kernel(sigma_blur, kernel_size).to(device)
        filter_tensor = torch.zeros(
            (1, num_channels, dim_image, dim_image), device=device
        )
        filter_tensor[..., :kernel_size, :kernel_size] = self.kernel
        self.filter = torch.roll(
            filter_tensor,
            shifts=(-(kernel_size-1)//2, -(kernel_size-1)//2),
            dims=(2, 3),
        )
        self.device = device

    def H(self, x):
        if self.mode != "fft":
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(
                x, kernel, stride=1, padding="same", groups=x.shape[1]
            )
        return torch.real(torch.fft.ifft2(
            torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.filter)
        ))

    def H_adj(self, x):
        if self.mode != "fft":
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(
                x, kernel, stride=1, padding="same", groups=x.shape[1]
            )
        return torch.real(torch.fft.ifft2(
            torch.fft.fft2(x.to(self.device))
            * torch.conj(torch.fft.fft2(self.filter))
        ))


class Superresolution(Degradation):
    def __init__(self, sf, dim_image, mode=None, device="cuda"):
        super().__init__()
        self.sf = sf
        self.mode = mode
        self.downsampling_matrix = create_downsampling_matrix(
            dim_image, dim_image, sf, device
        )

    def H(self, x):
        return downsample(x, self.sf)

    def H_adj(self, x):
        return upsample(x, self.sf)
