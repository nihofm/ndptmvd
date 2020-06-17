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
# CMD LINE SETTINGS

cmdline = argparse.ArgumentParser(description='Train a denoising autoencoder')
# required args
cmdline.add_argument('x_train', help='path to directory containing input training data')
cmdline.add_argument('y_train', help='path to directory containing target training data')
cmdline.add_argument('x_val', help='path to directory containing input validation data')
cmdline.add_argument('y_val', help='path to directory containing target validation data')
# optional args
cmdline.add_argument('-t', '--type', type=str, default='image', help='type of network to train: [image, dual, temporal]')
cmdline.add_argument('-n', '--name', type=str, default='default', help='name of trained model')
cmdline.add_argument('-e', '--epochs', type=int, default=250, help='number of epochs to train')
cmdline.add_argument('-b', '--batch_size', type=int, default=4, help='mini batch size')
cmdline.add_argument('-w', '--workers', type=int, default=4, help='num workers to spawn for data loading')
cmdline.add_argument('-f','--features', type=int, default=[], nargs='*', help='Tuples of feature channels to select from input')
cmdline.add_argument('-l','--loss', type=str, default='hfs', help='type of loss function to use: [ssim, l1ssim, l2ssim, hfen, hfs]')
cmdline.add_argument('-lr', '--learning_rate', type=float, default=2e-4, help='adam hyperparameter: learning rate')
cmdline.add_argument('-b1', '--beta1', type=float, default=0.9, help='adam hyperparameter: beta1')
cmdline.add_argument('-b2', '--beta2', type=float, default=0.999, help='adam hyperparameter: beta2')
cmdline.add_argument('-c', '--clip', type=float, default=1, help='value for gradient value/norm clipping')
cmdline.add_argument('--big', action="store_true", help="use big dual autoencoder variant")
cmdline.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
cmdline.add_argument('--amp', action="store_true", help="use automatic mixed precision (fp16/fp32)")

# -----------------------------------------------------------
# FUNCS

def train_epoch(device, model, data_loader, optimizer, loss_fn, epoch):
    model.train()
    tq = tqdm.tqdm(total=len(data_loader)*args.batch_size)
    tq.set_description(f'Train: Epoch {epoch:4}, LR: {optimizer.param_groups[0]["lr"]:0.6f}')
    train_loss, train_ssim, train_psnr = 0, 0, 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_fn(prediction, target)
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item() * (1/len(data_loader))
            if 'temp' in args.type: prediction, target = prediction[:, prediction.size(1)//2].squeeze(1), target[:, target.size(1)//2].squeeze(1)
            train_ssim += losses.ssim(prediction, target).item() * (1/len(data_loader))
            train_psnr += losses.psnr(prediction, target).item() * (1/len(data_loader))
        tq.update(args.batch_size)
        tq.set_postfix(loss=f'{train_loss*len(data_loader)/(batch_idx+1):4.6f}',
                ssim=f'{train_ssim*len(data_loader)/(batch_idx+1):.4f}',
                psnr=f'{train_psnr*len(data_loader)/(batch_idx+1):4.4f}')
    tq.close()
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('SSIM/train', train_ssim, epoch)
    writer.add_scalar('PSNR/train', train_psnr, epoch)

def eval_epoch(device, model, data_loader, loss_fn, epoch):
    model.eval()
    tq = tqdm.tqdm(total=len(data_loader))
    tq.set_description(f'Test:  Epoch {epoch:4}')
    eval_loss, eval_ssim, eval_psnr = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            eval_loss += loss_fn(prediction, target).item() * (1/len(data_loader))
            if 'temp' in args.type: prediction, target = prediction[:, prediction.size(1)//2].squeeze(1), target[:, target.size(1)//2].squeeze(1)
            eval_ssim += losses.ssim(prediction, target).item() * (1/len(data_loader))
            eval_psnr += losses.psnr(prediction, target).item() * (1/len(data_loader))
            tq.update()
            tq.set_postfix(loss=f'{eval_loss*len(data_loader)/(batch_idx+1):4.6f}',
                    ssim=f'{eval_ssim*len(data_loader)/(batch_idx+1):.4f}',
                    psnr=f'{eval_psnr*len(data_loader)/(batch_idx+1):4.4f}')
    tq.close()
    writer.add_scalar('Loss/test', eval_loss, epoch)
    writer.add_scalar('SSIM/test', eval_ssim, epoch)
    writer.add_scalar('PSNR/test', eval_psnr, epoch)
    if epoch % 10 == 0:
        if 'temp' in args.type: data = data[:, data.size(1)//2].squeeze(1)
        writer.add_image(f'Prediction/test', torch.clamp(torch.cat((data[-1, 0:3], prediction[-1], target[-1]), dim=-1), 0, 1), epoch, dataformats='CHW')
    return eval_loss

# -----------------------------------------------------------
# MAIN

if __name__ == "__main__":

    # parse command line
    args = cmdline.parse_args()
    args.name = args.type + '_' + args.loss + '_' + args.name
    print('args:', args)

    # check for GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'Num GPUs: {torch.cuda.device_count()}')

    # load data set and setup data loader
    if 'image' in args.type:
        data_train = data.DataSetTrain(args.x_train, args.y_train, args.features)
        data_test = data.DataSetTest(args.x_val, args.y_val, args.features)
        model = models.Autoencoder(data_train.input_channels).to(device)
    elif 'dual' in args.type:
        data_train = data.DataSetTrain(args.x_train, args.y_train)
        data_test = data.DataSetTest(args.x_val, args.y_val)
        if args.big:
            model = models.AutoencoderDualF24Big().to(device)
        else:
            model = models.AutoencoderDualF24().to(device)
    elif 'temp' in args.type:
        data_train = data.DataSetTrainTemporal(args.x_train, args.y_train)
        data_test = data.DataSetTest(args.x_val, args.y_val, temporal=True)
        model = models.TemporalAdapter(models.AutoencoderDualF24()).to(device)
    else:
        raise ValueError(f'ERROR: Unsupported network/data type: {args.type}')

    # setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    if 'ssim' in args.loss:
        loss_fn = losses.SSIMLoss()
    elif 'l1ssim' in args.loss:
        loss_fn = losses.L1SSIMLoss()
    elif 'l2ssim' in args.loss:
        loss_fn = losses.L2SSIMLoss()
    elif 'hfen' in args.loss:
        loss_fn = losses.L1HFENLoss()
    elif 'hfs' in args.loss:
        loss_fn = losses.L1HFENSpecLoss()
    elif 'l1spec' in args.loss:
        loss_fn = losses.L1SpecLoss()
    elif 'temp' in args.loss:
        loss_fn = losses.L1HFENTemporalLoss()
    elif 'l1' in args.loss:
        loss_fn = torch.nn.L1Loss()
    elif 'l2' in args.loss:
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f'ERROR: Unsupported loss function type: {args.loss}')

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

    # init data loader
    def seed_fn(id): np.random.seed()
    train_loader = torch.utils.data.DataLoader(data_train, num_workers=args.workers, pin_memory=True, batch_size=args.batch_size, worker_init_fn=seed_fn)
    test_loader = torch.utils.data.DataLoader(data_test, num_workers=2, pin_memory=True, batch_size=max(1, torch.cuda.device_count()))

    # init tensorboard logger
    writer = SummaryWriter(comment='_' + args.name)

    # run training
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    print(f'Training model {args.name} of type {args.type} with batch size {args.batch_size} for {args.epochs} epochs.')
    print(f'Trainable params: {sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])}')
    parallel_model = torch.nn.DataParallel(model)

    for epoch in range(args.epochs):
        train_epoch(device, parallel_model, train_loader, optimizer, loss_fn, epoch)
        eval_loss = eval_epoch(device, parallel_model, test_loader, loss_fn, epoch)
        scheduler.step(eval_loss)
        # save checkpoint?
        if eval_loss < best_loss:
            best_loss = eval_loss
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
