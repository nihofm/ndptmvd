# external imports
import os
import time
import tqdm
import torch
import argparse
import imageio_ffmpeg
import numpy as np
# internal imports
import data
import models

# -----------------------------------------------------------
# CMD LINE SETTINGS

cmdline = argparse.ArgumentParser(description='Use a denoising autoencoder')
# required args
cmdline.add_argument('model', help='path to h5 model file')
cmdline.add_argument('input_data', help='path to directory containing input data')
cmdline.add_argument('target_data', help='path to directory containing target data')
# optional args
cmdline.add_argument('-t', '--type', type=str, default='image', help='type of dataset to feed: [image, reproj]')
cmdline.add_argument('-b', '--batch_size', type=int, default=1, help='mini batch size')
cmdline.add_argument('--images', action="store_true", help="store images")
cmdline.add_argument('--cmp', action="store_true", help="write comparison images")
cmdline.add_argument('--format', type=str, default='jpg', help='output image format')
cmdline.add_argument('--fps', type=int, default=24, help='output video frame rate')
cmdline.add_argument('-f','--features', type=int, default=[], nargs='*', help='Tuples of feature channels to select from input')

# -----------------------------------------------------------
# MAIN

if __name__ == "__main__":

    # parse command line
    args = cmdline.parse_args()
    print('args:', args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup model
    model = models.load(args.model)
    model.eval()
    model.to(device)
    # model = torch.nn.DataParallel(model)

    # setup data set
    if 'reproj' in args.type:
        dataset = data.DataSetTestReproj(args.input_data, args.target_data)
    else:
        dataset = data.DataSetTest(args.input_data, args.target_data, args.features)
    # FIXME multiprocess loading with proper ordering?
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=1, pin_memory=True, batch_size=args.batch_size)

    # setup files
    name = os.path.splitext(os.path.basename(args.model))[0] + '--' + os.path.basename(os.path.dirname(args.input_data + '/'))
    os.makedirs(name, exist_ok=True)

    # setup video storage
    (sample, _) = dataset[0]
    video = np.empty((len(dataset), sample.shape[-2], 3*sample.shape[-1], 3), dtype='uint8')

    times = []

    # print('Denoising...')
    with torch.no_grad():
        tq = tqdm.tqdm(total=len(data_loader)*args.batch_size, desc='Denoise')
        for idx, (in_data, target) in enumerate(data_loader):
            in_data, target = in_data.to(device), target.to(device)
            torch.cuda.synchronize()
            start = time.time()
            prediction = model(in_data)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            # store video frame(s)
            x = in_data[:, data.N_FRAMES, 0:3, :, :] if 'reproj' in args.type else in_data[:, 0:3, :, :]
            p = prediction[:, 0:3, :, :]
            y = target[:, 0:3, :, :]
            # postprocess
            frame = (torch.sqrt(torch.clamp(torch.cat((x, p, y), dim=-1), 0, 1)) * 255).to(torch.uint8)
            frame = frame.transpose_(-3, -1).transpose_(-3, -2)
            video[args.batch_size*idx:args.batch_size*idx+p.size(0)] = frame.cpu().numpy()
            # write images to disk?
            if args.images:
                img = torch.cat((x, p, y), dim=-1) if args.cmp else p
                data.write([f'{name}/{name}_{args.batch_size*idx+j:06}.{args.format}' for j in range(frame.size(0))], img.cpu().numpy())
            tq.update(args.batch_size)
        tq.close()

    print(f'avg inference time (in s):', np.array(times).mean(), 'std:', np.array(times).std())

    # write video
    ffmpeg = imageio_ffmpeg.write_frames(f'{name}/{name}.mp4', (3*sample.shape[-1], sample.shape[-2]), fps=args.fps, quality=5)
    ffmpeg.send(None) # seed
    ffmpeg.send(video)
    ffmpeg.close()
    print(f'{name}/{name}.mp4 written.')
    # make sure images were written
    data.pool.close()
    data.pool.join()
