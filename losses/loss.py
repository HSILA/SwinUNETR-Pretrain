# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur

def compute_curvature_isophotes_batch(images, sigma=1):
    """
    Computes the mean curvature of isophotes for a batch of 3D channel images using PyTorch.
    
    :param images: A batch of 3D images stored in a PyTorch tensor of shape (batch_size, dim, height, width, depth).
    :param sigma: Standard deviation for Gaussian blur.
    :return: A tensor of the same shape as 'images' containing the curvature of isophotes for each image.
    """
    images = images.float()

    kernel_size = int(6*sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    L = gaussian_blur(images.squeeze(1), kernel_size=(
        kernel_size, kernel_size), sigma=(sigma, sigma)).unsqueeze(1)

    L_x = torch.gradient(L, dim=4)[0]
    L_y = torch.gradient(L, dim=3)[0]
    L_z = torch.gradient(L, dim=2)[0]

    L_xx = torch.gradient(L_x, axis=4)[0]
    L_yy = torch.gradient(L_y, axis=3)[0]
    L_zz = torch.gradient(L_z, axis=2)[0]
    L_xy = torch.gradient(L_x, axis=3)[0]
    L_zx = torch.gradient(L_z, axis=4)[0]
    L_zy = torch.gradient(L_z, axis=3)[0]

    numerator = (L_x**2) * (L_yy + L_zz) - 2 * L_x * L_y * L_xy \
        + (L_y**2) * (L_xx + L_zz) - 2 * L_y * L_z * L_zy \
        + (L_z**2) * (L_xx + L_yy) - 2 * L_z * L_x * L_zx

    denom = (L_x**2 + L_y**2 + L_z**2)**(3/2)

    k = numerator / denom

    k[torch.isnan(k)] = 0
    k[torch.isinf(k)] = 0

    return k

class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0
        self.alpha4 = args.alpha4

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        if self.alpha4 == 0:
            mci_loss = torch.tensor(0.0)
        else:
            mci_loss = self.alpha4 * \
                self.recon_loss(
                    compute_curvature_isophotes_batch(output_recons),
                    compute_curvature_isophotes_batch(target_recons))
        total_loss = rot_loss + contrast_loss + recon_loss + mci_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss, mci_loss)
