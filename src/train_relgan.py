# external imports
import os
import math
import tqdm
import torch
import torchvision
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# internal imports
import data
import losses
import models

# -----------------------------------------------------------
# SETTINGS

cmdline = argparse.ArgumentParser(description='Train a reprojection denoising autoencoder')
# required args
cmdline.add_argument('x_train', help='path to directory containing input training data')
cmdline.add_argument('y_train', help='path to directory containing target training data')
cmdline.add_argument('x_val', help='path to directory containing input validation data')
cmdline.add_argument('y_val', help='path to directory containing target validation data')
# optional args
cmdline.add_argument('-n', '--name', type=str, default='default', help='name of trained model')
cmdline.add_argument('-e', '--epochs', type=int, default=250, help='number of epochs to train')
cmdline.add_argument('-b', '--batch_size', type=int, default=4, help='mini batch size')
cmdline.add_argument('-w', '--workers', type=int, default=4, help='num workers to spawn for data loading')
cmdline.add_argument('-lr_g', '--lr_generator', type=float, default=2e-4, help='hyperparameter: learning rate')
cmdline.add_argument('-lr_d', '--lr_discriminator', type=float, default=2e-4, help='hyperparameter: learning rate')
cmdline.add_argument('-b1', '--beta1', type=float, default=0.5, help='adam hyperparameter: beta1')
cmdline.add_argument('-b2', '--beta2', type=float, default=0.999, help='adam hyperparameter: beta2')
cmdline.add_argument('-c', '--clip', type=float, default=1, help='value for gradient norm clipping')
cmdline.add_argument('--generator', type=str, default=None, help='load generator weights from checkpoint')
cmdline.add_argument('--discriminator', type=str, default=None, help='load discriminator weights from checkpoint')
cmdline.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
cmdline.add_argument('--reproj', action="store_true", help="use reprojection autoencoder (feature select variant)")
cmdline.add_argument('--big', action="store_true", help="use big generator and discriminator variants")

# -----------------------------------------------------------
# MAIN

if __name__ == "__main__":

    # parse command line
    args = cmdline.parse_args()
    args.name = 'relgan_' + args.name
    print('args:', args)

    # check for GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'Num GPUs: {torch.cuda.device_count()}')

    # load data set and setup data loader
    data_train = data.DataSetTrainReproj(args.x_train, args.y_train) if args.reproj else data.DataSetTrain(args.x_train, args.y_train)
    data_test = data.DataSetTestReproj(args.x_val, args.y_val) if args.reproj else data.DataSetTest(args.x_val, args.y_val)
    def seed_fn(id): np.random.seed()
    train_loader = torch.utils.data.DataLoader(data_train, num_workers=args.workers, pin_memory=True, batch_size=args.batch_size, worker_init_fn=seed_fn)
    test_loader = torch.utils.data.DataLoader(data_test, num_workers=2, pin_memory=True, batch_size=max(1, torch.cuda.device_count()))

    # setup generator model
    modelG = models.AutoencoderDualF24() if not args.big else models.AutoencoderDualF24Big()
    if args.reproj: modelG = models.ReprojFeatureSelectAdapter(modelG, data_train.input_channels, data.N_FRAMES)
    if args.generator: modelG.load_state_dict(torch.load(args.generator)['weights'])
    modelG.to(device)
    # setup discriminator model
    modelD = models.Discriminator() if not args.big else models.Discriminator256x3()
    if args.discriminator: modelD.load_state_dict(torch.load(args.discriminator)['weightsD'])
    modelD.to(device)
    # setup optimizers and schedulers
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=args.lr_generator, betas=(args.beta1, args.beta2))
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=args.lr_discriminator, betas=(args.beta1, args.beta2))
    schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, patience=25, cooldown=10, factor=0.5, min_lr=1e-5, threshold=1e-5)
    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, patience=25, cooldown=10, factor=0.5, min_lr=1e-5, threshold=1e-5)
    # setup reconstruction loss function
    loss_fn = losses.L1SpecLoss()

    # load checkpoint?
    best_loss = float('inf')
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        best_loss = state['best_loss']
        modelG.load_state_dict(state['weights'])
        optimizerG.load_state_dict(state['optimizer'])
        modelD.load_state_dict(state['weightsD'])
        optimizerD.load_state_dict(state['optimizerD'])

    # init tensorboard logger
    writer = SummaryWriter(comment='_' + args.name)

    # run training
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    print(f'Training relgan model {args.name} with batch size {args.batch_size} for {args.epochs} epochs.')
    print(f'Trainable params generator: {sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, modelG.parameters())])}')
    print(f'Trainable params discriminator: {sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, modelD.parameters())])}')
    modelG_p = torch.nn.DataParallel(modelG)
    modelD_p = torch.nn.DataParallel(modelD)

    for epoch in range(args.epochs):

        # -----------------------------------------
        # train for one epoch

        modelG_p.train()
        tq = tqdm.tqdm(total=len(train_loader)*args.batch_size)
        tq.set_description(f'Train: Epoch {epoch:4}, LR: {optimizerG.param_groups[0]["lr"]:0.6f}')
        train_loss, train_ssim, train_psnr = 0, 0, 0
        for batch_idx, (x, y) in enumerate(train_loader):

            # run forward
            x, y = x.to(device), y.to(device)
            p = modelG_p(x)
            if batch_idx % 10 == 0:
                torchvision.utils.save_image(F.interpolate(p.detach(), size=256), f'p_{args.name}.jpg', nrow=4)

            # train discriminator
            modelD_p.train()
            optimizerD.zero_grad()
            C_real = modelD_p(y)
            C_fake = modelD_p(torch.clamp(p.detach(), 0, 1))
            mean_C_real = torch.mean(C_real, dim=(0,), keepdim=True).expand_as(C_real).detach()
            mean_C_fake = torch.mean(C_fake, dim=(0,), keepdim=True).expand_as(C_fake).detach()
            loss1 = F.mse_loss(C_real - mean_C_fake, torch.tensor(1.0).to(device).expand_as(C_real))
            loss2 = F.mse_loss(C_fake - mean_C_real, torch.tensor(-1.0).to(device).expand_as(C_fake))
            lossDD = 0.5 * (loss1 + loss2)
            lossDD.backward()
            torch.nn.utils.clip_grad_value_(modelD.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(modelD.parameters(), args.clip)
            optimizerD.step()

            # train generator
            modelD_p.eval()
            optimizerG.zero_grad()
            C_real = modelD_p(y)
            C_fake = modelD_p(torch.clamp(p, 0, 1))
            mean_C_real = torch.mean(C_real, dim=(0,), keepdim=True).expand_as(C_real).detach()
            mean_C_fake = torch.mean(C_fake, dim=(0,), keepdim=True).expand_as(C_fake).detach()
            loss1 = F.mse_loss(C_fake - mean_C_real, torch.tensor(1.0).to(device).expand_as(C_fake))
            loss2 = F.mse_loss(C_real - mean_C_fake, torch.tensor(-1.0).to(device).expand_as(C_real))
            lossDG = 0.5 * (loss1 + loss2)
            lossR = loss_fn(p, y)
            # loss = (2 * lossDG * lossR) / (lossDG + lossR)
            loss = 0.6 * lossDG + 0.4 * lossR
            loss.backward()
            torch.nn.utils.clip_grad_value_(modelG.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(modelG.parameters(), args.clip)
            optimizerG.step()

            with torch.no_grad():
                train_loss += loss.item() * (1/len(train_loader))
                train_ssim += losses.ssim(p, y).item() * (1/len(train_loader))
                train_psnr += losses.psnr(p, y).item() * (1/len(train_loader))
            tq.update(args.batch_size)
            tq.set_postfix(loss=f'{train_loss*len(train_loader)/(batch_idx+1):4.6f}',
                    ssim=f'{train_ssim*len(train_loader)/(batch_idx+1):.4f}',
                    psnr=f'{train_psnr*len(train_loader)/(batch_idx+1):4.4f}')
        tq.close()
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('SSIM/train', train_ssim, epoch)
        writer.add_scalar('PSNR/train', train_psnr, epoch)

        # -----------------------------------------
        # test for one epoch

        modelG_p.eval()
        tq = tqdm.tqdm(total=len(test_loader))
        tq.set_description(f'Test:  Epoch {epoch:4}')
        eval_loss, eval_ssim, eval_psnr = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                prediction = modelG_p(x)
                eval_loss += loss_fn(prediction, y).item() * (1/len(test_loader))
                eval_ssim += losses.ssim(prediction, y).item() * (1/len(test_loader))
                eval_psnr += losses.psnr(prediction, y).item() * (1/len(test_loader))
                tq.update()
                tq.set_postfix(loss=f'{eval_loss*len(test_loader)/(batch_idx+1):4.6f}',
                        ssim=f'{eval_ssim*len(test_loader)/(batch_idx+1):.4f}',
                        psnr=f'{eval_psnr*len(test_loader)/(batch_idx+1):4.4f}')
        tq.close()
        writer.add_scalar('Loss/test', eval_loss, epoch)
        writer.add_scalar('SSIM/test', eval_ssim, epoch)
        writer.add_scalar('PSNR/test', eval_psnr, epoch)
        if epoch % 10 == 0:
            if args.reproj: x = x[:, data.N_FRAMES]
            writer.add_image(f'Prediction/test', torch.clamp(torch.cat((x[-1, 0:3], prediction[-1], y[-1]), dim=-1), 0, 1), epoch, dataformats='CHW')

        # -----------------------------------------
        # run learning rate scheduler

        schedulerG.step(eval_loss)
        schedulerD.step(eval_loss)

        # -----------------------------------------
        # save best model

        if eval_loss < best_loss:
            best_loss = eval_loss
            filename = os.path.join('checkpoints', args.name) + '.pt'
            checkpoint = {
                'best_loss': best_loss,
                'weights': modelG.state_dict(),
                'optimizer': optimizerG.state_dict(),
                'weightsD': modelD.state_dict(),
                'optimizerD': optimizerD.state_dict(),
            }
            torch.save(checkpoint, filename)
            torch.save(modelG, os.path.join('models', args.name) + '.pt')
            print(f'Checkpoint saved: {filename} (loss: {best_loss:.4f})')
