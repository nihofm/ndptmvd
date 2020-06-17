# external imports
import os
import math
import tqdm
import torch
import argparse
import numpy as np
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
# optional args
cmdline.add_argument('-n', '--name', type=str, default='default', help='name of trained model')
cmdline.add_argument('-e', '--epochs', type=int, default=250, help='number of epochs to train')
cmdline.add_argument('-b', '--batch_size', type=int, default=4, help='mini batch size')
cmdline.add_argument('-w', '--workers', type=int, default=4, help='num workers to spawn for data loading')
cmdline.add_argument('-lr', '--learning_rate', type=float, default=2e-4, help='adam hyperparameter: learning rate')
cmdline.add_argument('-b1', '--beta1', type=float, default=0.9, help='adam hyperparameter: beta1')
cmdline.add_argument('-b2', '--beta2', type=float, default=0.999, help='adam hyperparameter: beta2')
cmdline.add_argument('-c', '--clip', type=float, default=1, help='value for gradient value/norm clipping')
cmdline.add_argument('--feature_select', action="store_true", help="use feature select variant")
cmdline.add_argument('--autoencoder', type=str, default=None, help='load autoencoder from checkpoint')
cmdline.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
cmdline.add_argument('--amp', action="store_true", help="use automatic mixed precision (fp16/fp32)")

# -----------------------------------------------------------
# MAIN

if __name__ == "__main__":

    # parse command line
    args = cmdline.parse_args()
    args.name = 'reproj_' + args.name
    print('args:', args)

    # check for GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'Num GPUs: {torch.cuda.device_count()}')

    # load data set and setup data loader
    data_train = data.DataSetTrainReproj(args.x_train, args.y_train)
    def seed_fn(id): np.random.seed()
    train_loader = torch.utils.data.DataLoader(data_train, num_workers=args.workers, pin_memory=True, batch_size=args.batch_size, worker_init_fn=seed_fn)

    # load and setup model
    autoencoder = models.AutoencoderDualF24()
    if args.autoencoder: autoencoder.load_state_dict(torch.load(args.autoencoder)['weights'])
    if args.feature_select:
        model = models.ReprojFeatureSelectAdapter(autoencoder, data_train.input_channels, data.N_FRAMES)
    else:
        model = models.ReprojSelectAdapter(autoencoder, data_train.input_channels, data.N_FRAMES)
    model.to(device)

    # setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    loss_fn = losses.L1HFENSpecLoss()

    # init mixed precision training?
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # LR scheduler (needs init after amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, cooldown=10, factor=0.5, min_lr=1e-5, threshold=1e-5)

    # load checkpoint?
    best_loss = float('inf')
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        best_loss = state['best_loss']
        model.load_state_dict(state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        if args.amp: amp.load_state_dict(state['amp'])

    # init tensorboard logger
    writer = SummaryWriter(comment='_' + args.name)

    # run training
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    print(f'Training model {args.name} with batch size {args.batch_size} for {args.epochs} epochs.')
    print(f'Trainable params: {sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])}')
    parallel_model = torch.nn.DataParallel(model)

    for epoch in range(args.epochs):

        # -----------------------------------------
        # train for one epoch

        parallel_model.train()
        tq = tqdm.tqdm(total=len(train_loader)*args.batch_size)
        tq.set_description(f'Train: Epoch {epoch:4}, LR: {optimizer.param_groups[0]["lr"]:0.6f}')
        train_loss, train_ssim, train_psnr = 0, 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            prediction = parallel_model(data)
            loss = loss_fn(prediction, target)
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_value_(parallel_model.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), args.clip)
            optimizer.step()
            with torch.no_grad():
                train_loss += loss.item() * (1/len(train_loader))
                train_ssim += losses.ssim(prediction, target).item() * (1/len(train_loader))
                train_psnr += losses.psnr(prediction, target).item() * (1/len(train_loader))
            tq.update(args.batch_size)
            tq.set_postfix(loss=f'{train_loss*len(train_loader)/(batch_idx+1):4.6f}',
                    ssim=f'{train_ssim*len(train_loader)/(batch_idx+1):.4f}',
                    psnr=f'{train_psnr*len(train_loader)/(batch_idx+1):4.4f}')
        tq.close()
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('SSIM/train', train_ssim, epoch)
        writer.add_scalar('PSNR/train', train_psnr, epoch)
        scheduler.step(train_loss)

        # -----------------------------------------
        # save checkpoint for best loss

        if train_loss < best_loss:
            best_loss = train_loss
            filename = os.path.join('checkpoints', args.name) + '.pt'
            checkpoint = {
                'best_loss': best_loss,
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if args.amp: checkpoint['amp'] = amp.state_dict()
            torch.save(checkpoint, filename)
            if not args.amp: # FIXME bugged with amp
                torch.save(model, os.path.join('models', args.name) + '.pt')
            print(f'Checkpoint saved: {filename} (loss: {best_loss:.6f})')
