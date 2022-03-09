from tokenize import group
import numpy as np
import math
import torch
import torch.nn.functional as F
import typing


def get_gausskernel_size(sigma, force_odd=True):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    if ksize % 2 == 0 and force_odd:
        ksize += 1
    return int(ksize)


def gaussian1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Function that computes values of a (1D) Gaussian with zero mean and variance sigma^2"""
    # out =  torch.zeros(x.shape)
    return np.exp(-np.square(x) / (np.square(sigma) * 2)) / (np.sqrt(2 * np.pi) * sigma)
    # return out


def gaussian_deriv1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Function that computes values of a (1D) Gaussian derivative"""
    # out =  torch.zeros(x.shape)
    return np.multiply(-x / np.square(sigma), gaussian1d(x, sigma))
    # return out


def filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    # TODO can be rectangular

    pad_shape = np.ones((1, 4)) * np.ceil(kernel.size(dim=0) / 2)
    pad_shape = pad_shape.flatten().astype(int)
    pad_shape = (
        kernel.size(dim=1) // 2,
        kernel.size(dim=1) // 2,
        kernel.size(dim=0) // 2,
        kernel.size(dim=0) // 2,
    )
    out = F.pad(x, tuple(pad_shape), "replicate",)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat((x.size(1), 1, 1, 1))
    return F.conv2d(input=out, weight=kernel, groups=x.size(1))


def gaussian_filter2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.

    Arguments:
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    """
    ksize = get_gausskernel_size(sigma)
    # k_ones = np.ones((1, ksize))
    k_ones = torch.linspace(-(ksize - 1) / 2, (ksize - 1) / 2, ksize).unsqueeze(0)
    x_kernel = torch.tensor(gaussian1d(k_ones, sigma=sigma))
    y_kernel = x_kernel.T
    out = filter2d(x, x_kernel)
    out = filter2d(out, y_kernel)
    return out


def x_derivative_spatial(grid, x, sigma):
    x_kernel = torch.tensor(gaussian_deriv1d(grid, sigma=sigma))
    y_kernel = torch.tensor(gaussian1d(grid, sigma=sigma)).T
    x_grad_tmp = filter2d(x, x_kernel)
    x_grad = filter2d(x_grad_tmp, y_kernel)
    return x_grad


def y_derivative_spatial(grid, x, sigma):
    y_kernel = torch.tensor(gaussian_deriv1d(grid, sigma=sigma)).T
    x_kernel = torch.tensor(gaussian1d(grid, sigma=sigma))
    y_grad_tmp = filter2d(x, x_kernel)
    y_grad = filter2d(y_grad_tmp, y_kernel)
    return y_grad


def spatial_gradient_first_order(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """
    b, c, h, w = x.shape
    ksize = get_gausskernel_size(sigma)

    grid = torch.linspace(-(ksize - 1) / 2, (ksize - 1) / 2, ksize).unsqueeze(0)
    # x_kernel = torch.tensor(gaussian_deriv1d(grid, sigma=sigma))
    # y_kernel = torch.tensor(gaussian1d(grid, sigma=sigma)).T
    # x_grad = filter2d(x, x_kernel)
    # x_grad = filter2d(x_grad, y_kernel)
    # # y_kernel2 = x_kernel
    # x_kernel = torch.tensor(gaussian1d(grid, sigma=sigma))
    # y_kernel = torch.tensor(gaussian_deriv1d(grid, sigma=sigma)).T
    # y_grad = filter2d(x, y_kernel)
    # y_grad = filter2d(y_grad, x_kernel)

    x_grad = x_derivative_spatial(grid, x, sigma)
    y_grad = y_derivative_spatial(grid, x, sigma)
    out = torch.stack([x_grad, y_grad], 2)

    # out = torch.zeros(b, c, 2, h, w)
    return out


def spatial_gradient_second_order(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Computes the second order image derivative in xx, xy, yy directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 3, H, W)`

    """
    xd = torch.tensor([[0.5, 0, -0.5]]).float()

    ksize = get_gausskernel_size(sigma)
    grid = torch.linspace(-(ksize - 1) / 2, (ksize - 1) / 2, ksize).unsqueeze(0)
    x_grad = x_derivative_spatial(grid, x, sigma)
    y_grad = y_derivative_spatial(grid, x, sigma)
    xx_grad = filter2d(x_grad, xd)
    xy_grad = filter2d(x_grad, xd.T)
    yy_grad = filter2d(y_grad, xd.T)

    out = torch.stack([xx_grad, xy_grad, yy_grad], 2)
    return out


def affine(center: torch.Tensor, unitx: torch.Tensor, unity: torch.Tensor) -> torch.Tensor:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image

    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 2)`, :math:`(B, 2)`, :math:`(B, 2)`
        - Output: :math:`(B, 3, 3)`

    """
    assert center.size(0) == unitx.size(0)
    assert center.size(0) == unity.size(0)
    B = center.size(0)
    coords = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
    # coords = coords.unsqueeze(0).repeat(B, 1, 1)
    p = torch.stack([center, unitx, unity], 2)
    o = torch.ones((B, 1, 3))
    p = torch.cat((p, o), 1)
    print(p.shape)
    print(o.shape)

    t = np.dot(p, np.linalg.inv(coords))
    # out = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    return torch.tensor(t).float()


def extract_affine_patches(
    input: torch.Tensor, A: torch.Tensor, img_idxs: torch.Tensor, PS: int = 32, ext: float = 6.0
):
    """Extract patches defined by affine transformations A from image tensor X.

    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors.

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    b, ch, h, w = input.size()
    num_patches = A.size(0)

    foo = torch.linspace(-ext, ext, PS)
    yy, xx = torch.meshgrid(foo, foo)
    xr = torch.reshape(xx, (-1, PS * PS))
    yr = torch.reshape(yy, (-1, PS * PS))
    zr = torch.ones_like(xr)
    pts = torch.cat([xr, yr, zr], 0)
    # A[:, 0, :] /= w
    # A[:, 1, :] /= h

    pts_t = torch.matmul(A, pts)
    pts_t[:, 0, :] = pts_t[:, 0, :] * 2 / w - 1.0
    pts_t[:, 1, :] = pts_t[:, 1, :] * 2 / h - 1.0
    pts_resh = torch.reshape(pts_t, (-1, 3, PS, PS))[:, 0:2, ...]
    pts_perm = pts_resh.permute((0, 2, 3, 1))  #!!
    out = [
        F.grid_sample(
            input[img_idxs[i]].unsqueeze(0).float(), pts_perm[i, ...].unsqueeze(0).float()
        )
        for i in range(num_patches)
    ]
    if len(out) > 0:
        return torch.cat(out, 0)
    else:
        return torch.tensor([]).float()

    # return out.double()
    # return out


def extract_antializased_affine_patches(
    input: torch.Tensor, A: torch.Tensor, img_idxs: torch.Tensor, PS: int = 32, ext: float = 6.0
):
    """Extract patches defined by affine transformations A from scale pyramid created image tensor X.
    It runs your implementation of the `extract_affine_patches` function, so it would not work w/o it.

    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors.

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    import kornia

    b, ch, h, w = input.size()
    num_patches = A.size(0)
    scale = (kornia.feature.get_laf_scale(ext * A.unsqueeze(0)[:, :, :2, :]) / float(PS))[0]
    half: float = 0.5
    pyr_idx = (scale.log2()).relu().long()
    cur_img = input
    cur_pyr_level = 0
    out = torch.zeros(num_patches, ch, PS, PS).to(device=A.device, dtype=A.dtype)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        _, ch_cur, h_cur, w_cur = cur_img.size()
        scale_mask = (pyr_idx == cur_pyr_level).squeeze()
        if (scale_mask.float().sum()) >= 0:
            scale_mask = (scale_mask > 0).view(-1)
            current_A = A[scale_mask]
            current_A[:, :2, :3] *= float(h_cur) / float(h)
            patches = extract_affine_patches(cur_img, current_A, img_idxs[scale_mask], PS, ext)
            out.masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.geometry.pyrdown(cur_img)
        cur_pyr_level += 1
    return out
