import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

from model import Generator, Discriminator
from config import get_config, set_config
from utils import save_state, load_state
from dataset import load_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str)
    parser.add_argument('--data_path', type=str, default='data')
    args = parser.parse_args()

    # Load Configuration
    set_config(args.yaml_path)
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    train_data, train_loader = load_dataset(args.data_path)

    # Build Model
    G = Generator().to(device)
    D = Discriminator().to(device)
    G_optim = optim.Adam(G.parameters(),
                         lr=cfg['TRAINING']['LEARNING_RATE'],
                         betas=(cfg['TRAINING']['BETA1'], cfg['TRAINING']['BETA2']))
    D_optim = optim.Adam(D.parameters(),
                         lr=cfg['TRAINING']['LEARNING_RATE'],
                         betas=(cfg['TRAINING']['BETA1'], cfg['TRAINING']['BETA2']))

    if os.path.exists(os.path.join('save', cfg['NAME'], 'state.pth')):
        current_iter, G_losses, D_losses = load_state(G, D, G_optim, D_optim)
    else:
        current_iter = 0
        G_losses = []
        D_losses = []

    # Multi GPU
    if torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    fixed_noise = torch.rand(25, cfg['MODEL']['Z_DIM'], 1, 1, device=device) * 2 - 1  # [-1, 1]

    real_label = torch.ones(cfg['TRAINING']['BATCH_SIZE'], device=device)  # 1
    fake_label = torch.zeros(cfg['TRAINING']['BATCH_SIZE'], device=device)  # 0
    criterion = nn.BCELoss()

    total_epoch = (cfg['TRAINING']['TOTAL_ITER'] - current_iter) // len(train_loader) + 1
    for epoch in range(total_epoch):
        pbar = tqdm(train_loader)
        for i, (real_data, _) in enumerate(pbar):
            real_data = real_data.to(device)
            bs = real_data.size(0)

            # Update D: maximize log(D(x)) + log(1 - D(G(z)))

            # Real Data
            D.zero_grad()
            pred_real_label = D(real_data)
            pred_real_label = pred_real_label.squeeze()  # [batch_size, 1, 1, 1] -> [batch_size]

            D_real_loss = criterion(pred_real_label, real_label[:bs])
            D_real_loss.backward()

            # Fake Data
            z_noise = torch.rand(bs, cfg['MODEL']['Z_DIM'], 1, 1, device=device) * 2 - 1  # [-1, 1]
            fake_img = G(z_noise)
            pred_fake_label = D(fake_img)
            pred_fake_label = pred_fake_label.squeeze()

            D_fake_loss = criterion(pred_fake_label, fake_label[:bs])
            D_fake_loss.backward()

            D_loss = D_real_loss + D_fake_loss
            D_optim.step()

            # Update G: maximize log(D(G(z)))
            G.zero_grad()
            z_noise = torch.rand(bs, cfg['MODEL']['Z_DIM'], 1, 1, device=device) * 2 - 1  # [-1, 1]
            fake_img = G(z_noise)
            pred_fake_label = D(fake_img)
            pred_fake_label = pred_fake_label.squeeze()

            G_loss = criterion(pred_fake_label, real_label[:bs])
            G_loss.backward()

            G_optim.step()

            pbar.set_description('epoch={:d}/{:d}, iter={:d}/{:d} D_loss={:.4f}, G_loss={:.4f}'.format(
                epoch + 1, total_epoch, current_iter, cfg['TRAINING']['TOTAL_ITER'], D_loss.item(), G_loss.item()
            ))

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            if current_iter % 500 == 0 or current_iter == cfg['TRAINING']['TOTAL_ITER']:
                fixed_img = G(fixed_noise).detach().cpu()
                fixed_img = vutils.save_image(
                    fixed_img,
                    os.path.join('result', cfg['NAME'], 'fixed_img_{}.png'.format(current_iter)),
                    padding=2, nrow=5, normalize=True)
                save_state(G, D, G_optim, D_optim, current_iter, G_losses, D_losses)
                if current_iter == cfg['TRAINING']['TOTAL_ITER']:
                    break

            current_iter += 1

        # end for dataloader
    # end for epoch