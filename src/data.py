import os
import h5py
import tqdm
import torch
import atexit
import imageio
import hashlib
import tempfile
import numpy as np
from multiprocessing.pool import ThreadPool

# -----------------------------------------------------------
# setup

TILE_SIZE = 256 # width and height of tiles
N_FRAMES = 1 # number of previous AND subsequent frames, i.e. (2*N_FRAMES)+1 in total
pool = ThreadPool(os.cpu_count())
atexit.register(pool.join)
atexit.register(pool.close)
imageio.plugins.freeimage.download() # hdr support

# -----------------------------------------------------------
# directory utility functions

def cache_path(path, ext):
    name = hashlib.md5(path.encode()).hexdigest()
    cache = os.path.join(os.path.join(tempfile.gettempdir(), 'dl-cache'), name)
    # cache = os.path.join(os.path.join(os.path.dirname(__file__), '../data-cache'), name)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    return os.path.splitext(cache)[0] + ext

def glob_directory(directory, key='.hdr', filter_feature_maps=True):
    in_files = (os.path.join(directory, x) for x in os.listdir(directory))
    in_files = filter(lambda x: key in x, in_files)
    if filter_feature_maps:
        in_files = filter(lambda x: '_pos' not in os.path.basename(x), in_files)
        in_files = filter(lambda x: '_norm' not in os.path.basename(x), in_files)
        in_files = filter(lambda x: '_alb' not in os.path.basename(x), in_files)
        in_files = filter(lambda x: '_vol' not in os.path.basename(x), in_files)
    return sorted(list(in_files))

def glob_directory_recursive(directory, key='.hdr', filter_feature_maps=True):
    files = glob_directory(directory, key, filter_feature_maps)
    for root, dirs, _ in os.walk(directory):
        for name in sorted(dirs):
            d = os.path.join(root, name)
            files.extend(glob_directory(d, key, filter_feature_maps))
    return files

# -----------------------------------------------------------
# image utility functions

def _load_image_uint8(filename):
    img = imageio.imread(filename)
    if img.dtype == 'float32' or img.dtype == 'float64':
        img = np.round(np.clip(img, 0, 1) * 255).astype('uint8')
    elif img.dtype == 'uint16':
        img = np.round(255 * img.astype('float32') / 65535).astype('uint8')
    assert img.dtype == 'uint8', "Unhandled image data type!"
    return np.transpose(img, [2, 0, 1]) # to channels first

def _load_image_float(filename):
    img = imageio.imread(filename)
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255
    elif img.dtype == 'uint16':
        img = img.astype('float32') / 65535
    assert img.dtype == 'float32', "Unhandled image data type!"
    return np.transpose(img, [2, 0, 1]) # to channels first

def _concat_feature_map(img, filename):
    return np.concatenate((img, _load_image_uint8(filename)), axis=-3) if os.path.isfile(filename) else img

def _load_image_internal(filename, load_features=True):
    """ load rgb image file from disk with feature maps and return numpy array with shape (c, h, w) """
    img = _load_image_uint8(filename)
    if load_features:
        # test if additional data available
        root, ext = os.path.splitext(filename)
        root = root.replace(ext, '') # fix double extension fuckup
        img = _concat_feature_map(img, root + '_pos' + ext)
        img = _concat_feature_map(img, root + '_norm' + ext)
        img = _concat_feature_map(img, root + '_alb' + ext)
        img = _concat_feature_map(img, root + '_vol' + ext)
        img = _concat_feature_map(img, root + '_norm1' + ext)
        img = _concat_feature_map(img, root + '_alb1' + ext)
        img = _concat_feature_map(img, root + '_vol1' + ext)
        img = _concat_feature_map(img, root + '_norm2' + ext)
        img = _concat_feature_map(img, root + '_alb2' + ext)
        img = _concat_feature_map(img, root + '_vol2' + ext)
    return img

def preprocess_image(x):
    # convert to float32
    if x.dtype == 'uint8':
        x = x.astype('float32') * (1/255)
    elif x.dtype == 'uint16':
        x = x.astype('float32') * (1/65535)
    elif x.dtype == 'float64':
        x = x.astype('float32')
    elif x.dtype == 'float32':
        x = x
    else:
        raise ValueError('Unkown image data type')
    # clip and gamma adjust
    x = np.clip(x, 0, 1)
    x[..., 0:3, :, :] = np.square(x[..., 0:3, :, :])
    return x

def postprocess_image(x):
    # gamma adjust
    x[..., 0:3, :, :] = np.sqrt(np.clip(x[..., 0:3, :, :], 0, 1))
    # to uint8
    if x.dtype == 'uint8':
        return np.clip(x, 0, 255)
    elif x.dtype == 'float32' or x.dtype == 'float64':
        return (np.clip(x, 0, 1) * 255).astype('uint8')
    else:
        raise ValueError('Unkown image data type')

def load_image(filename):
    """ load rgb image file from disk and return numpy array of floats in [0, 1] with shape (c, h, w) """
    return preprocess_image(_load_image_internal(filename))

def write_image(filename, image, channels=3):
    """ write rgb image (numpy array of floats in [0, 1] with shape (c, h, w)) to disk """
    imageio.imwrite(filename, np.transpose(postprocess_image(image[0:channels]), [1, 2, 0]))
    print(f'{filename} written.')

def write(filenames, images, channels=3):
    """ async write either single image or list of images to filenames """
    if type(filenames) is not list:
        write_image(filenames, images, channels)
    else:
        assert len(filenames) == len(images), "Data size mismatch!"
        pool.starmap_async(lambda f, i: write_image(f, i, channels), zip(filenames, images))

# -----------------------------------------------------------
# data augmentations in (h, w, c) layout

def random_crop(x, y):
    from_x = np.random.randint(0, x.shape[-1] - TILE_SIZE)
    from_y = np.random.randint(0, x.shape[-2] - TILE_SIZE)
    x = x[..., :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
    y = y[..., :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
    return x, y

def random_flip(x, y):
    x, y = np.copy(x), np.copy(y)
    # random flip horizontally/vertically
    if np.random.sample() < 0.5:
        x = np.flip(x, axis=-1)
        y = np.flip(y, axis=-1)
    if np.random.sample() < 0.5:
        x = np.flip(x, axis=-2)
        y = np.flip(y, axis=-2)
    # random flip color channels (input, target, albedo and BRDF)
    if np.random.sample() < 0.33:
        x[..., [0, 1], :, :] = x[..., [1, 0], :, :]
        if x.shape[-3] >= 12: x[..., [9, 10], :, :] = x[..., [10, 9], :, :]
        if x.shape[-3] >= 18: x[..., [15, 16], :, :] = x[..., [16, 15], :, :]
        if x.shape[-3] >= 21: x[..., [18, 19], :, :] = x[..., [19, 18], :, :]
        y[..., [0, 1], :, :] = y[..., [1, 0], :, :]
    if np.random.sample() < 0.33:
        x[..., [1, 2], :, :] = x[..., [2, 1], :, :]
        if x.shape[-3] >= 12: x[..., [10, 11], :, :] = x[..., [11, 10], :, :]
        if x.shape[-3] >= 18: x[..., [16, 17], :, :] = x[..., [17, 16], :, :]
        if x.shape[-3] >= 21: x[..., [19, 20], :, :] = x[..., [20, 19], :, :]
        y[..., [1, 2], :, :] = y[..., [2, 1], :, :]
    # random flip sequence direction
    if len(x.shape) > 3 and len(y.shape) > 3:
        if np.random.sample() < 0.5:
            x = np.flip(x, axis=-4)
            y = np.flip(y, axis=-4)
    return x, y

def augment(x, y):
    assert x.shape[-2:0] == y.shape[-2:0]
    x, y = random_crop(x, y)
    x, y = random_flip(x, y)
    return x, y

# -----------------------------------------------------------
# build memory mapped data set (hdf5)

def make_dataset(directory, load_features=True, key='.hdr'):
    filename = cache_path(directory, '.h5')
    if not os.path.isfile(filename):
        print(f'Building dataset from {directory} -> {filename} (this may take a while)...')
        with h5py.File(filename, 'w') as f:
            files = glob_directory_recursive(directory, key=key, filter_feature_maps=True)
            assert len(files) > 0, "Dataset is empty!"
            # load single image to determine shape
            tmp = _load_image_internal(files[0], load_features)
            shape = (len(files), tmp.shape[-3], tmp.shape[-2], tmp.shape[-1])
            # create dataset and load fill with images
            dset = f.create_dataset('data', shape=shape, dtype=tmp.dtype)
            tq = tqdm.tqdm(total=len(files))
            def helper(dset, idx, f):
                dset[idx] = _load_image_internal(f, load_features)
                tq.update(1)
            pool.starmap(lambda i, f: helper(dset, i, f), enumerate(files))
            tq.close()
    return h5py.File(filename, 'r')['data']

def make_dataset_pos(directory, shape, key='.dat'):
    filename = cache_path(directory, '_pos.h5')
    if not os.path.isfile(filename):
        print(f'Building pos dataset from {directory} -> {filename} (this may take a while)...')
        with h5py.File(filename, 'w') as f:
            files = glob_directory_recursive(directory, key=key, filter_feature_maps=False)
            assert len(files) > 0, "Dataset is empty!"
            # load single image to determine shape
            shape = (len(files), 3, shape[-2], shape[-1])
            # create dataset and load fill with images
            dset = f.create_dataset('data', shape=shape, dtype='float32')
            tq = tqdm.tqdm(total=len(files))
            def helper(dset, idx, f):
                dset[idx] = np.flip(np.fromfile(f, dtype='float32').reshape(shape[-1], shape[-2], 3).transpose(2, 0, 1), axis=-2)
                tq.update(1)
            pool.starmap(lambda i, f: helper(dset, i, f), enumerate(files))
            tq.close()
    return h5py.File(filename, 'r')['data']

# -----------------------------------------------------------
# load and parse matrix data from given directory

def load_matrices(directory, shape, key='_matrices.txt'):
    floats = pool.map(lambda f: [float(val) for val in open(f, 'r').read().split()], glob_directory_recursive(directory, key=key))
    return [parse_proj(f, shape) for f in floats]

# -----------------------------------------------------------
# image data sets

class DataSetPredict(torch.utils.data.Dataset):
    def __init__(self, dir_x, features=[], temporal=False):
        print(f'Loading predict dataset from {dir_x}...')
        self.data_x = make_dataset(dir_x, True)
        self.features = [(features[2*i], features[2*i+1]) for i in range(len(features)//2)]
        self.temporal = temporal
        self.input_channels = sum((b - a for (a, b) in self.features)) if len(self.features) else self.data_x.shape[-3]

    def select_features(self, x):
        if len(self.features):
            x = np.concatenate([x[..., lo:hi, :, :] for (lo, hi) in self.features], axis=-3)
        return x

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx, ...]
        x = self.select_features(x)
        if self.temporal: x = np.expand_dims(x, 0)
        return preprocess_image(x)

class DataSetTest(torch.utils.data.Dataset):
    def __init__(self, dir_x, dir_y, features=[], temporal=False):
        print(f'Loading test dataset from {dir_x} and {dir_y}...')
        self.data_x = make_dataset(dir_x, True)
        self.data_y = make_dataset(dir_y, False)
        assert self.data_x.shape[-2:0] == self.data_y.shape[-2:0], "Data set size mismatch!"
        assert len(self.data_x) == len(self.data_y), "Data set length mismatch!"
        self.features = [(features[2*i], features[2*i+1]) for i in range(len(features)//2)]
        self.temporal = temporal
        self.input_channels = sum((b - a for (a, b) in self.features)) if len(self.features) else self.data_x.shape[-3]

    def select_features(self, x):
        if len(self.features):
            x = np.concatenate([x[..., lo:hi, :, :] for (lo, hi) in self.features], axis=-3)
        return x

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x, y = self.data_x[idx, ...], self.data_y[idx, ...]
        x, y = self.select_features(x), y
        if self.temporal: x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
        return preprocess_image(x), preprocess_image(y)

class DataSetTrain(torch.utils.data.Dataset):
    def __init__(self, dir_x, dir_y, features=[]):
        print(f'Loading train dataset from {dir_x} and {dir_y}...')
        self.data_x = make_dataset(dir_x, True)
        self.data_y = make_dataset(dir_y, False)
        assert self.data_x.shape[-2:0] == self.data_y.shape[-2:0], "Data set size mismatch!"
        assert len(self.data_x) == len(self.data_y), "Data set length mismatch!"
        self.features = [(features[2*i], features[2*i+1]) for i in range(len(features)//2)]
        self.input_channels = sum((b - a for (a, b) in self.features)) if len(self.features) else self.data_x.shape[-3]

    def select_features(self, x):
        if len(self.features):
            x = np.concatenate([x[..., lo:hi, :, :] for (lo, hi) in self.features], axis=-3)
        return x

    def __len__(self):
        return 4 * len(self.data_x) # ~4 random crops per image per epoch

    def __getitem__(self, idx):
        f = np.random.randint(0, len(self.data_x))
        from_x = np.random.randint(0, self.data_x.shape[-1] - TILE_SIZE)
        from_y = np.random.randint(0, self.data_x.shape[-2] - TILE_SIZE)
        x = self.data_x[f, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
        y = self.data_y[f, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
        x, y = random_flip(x, y)
        x, y = self.select_features(x), y
        return preprocess_image(x.copy()), preprocess_image(y.copy())

# -----------------------------------------------------------
# image sequence data sets

class DataSetTestTemporal(torch.utils.data.Dataset):
    def __init__(self, dir_x, dir_y):
        print(f'Loading temporal test dataset from {dir_x} and {dir_y}...')
        self.data_x = make_dataset(dir_x, True)
        self.data_y = make_dataset(dir_y, False)
        assert self.data_x.shape[-2:0] == self.data_y.shape[-2:0], "Data set size mismatch!"
        assert len(self.data_x) == len(self.data_y), "Data set length mismatch!"
        self.input_channels = self.data_x.shape[-3]

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = np.zeros((2*N_FRAMES+1, self.data_x.shape[-3], self.data_x.shape[-2], self.data_x.shape[-1]), dtype=self.data_x.dtype)
        y = np.zeros((2*N_FRAMES+1, self.data_y.shape[-3], self.data_y.shape[-2], self.data_y.shape[-1]), dtype=self.data_y.dtype)
        x[N_FRAMES] = self.data_x[idx]
        y[N_FRAMES] = self.data_y[idx]
        for i in range(1, N_FRAMES+1):
            if idx - i >= 0:
                x[N_FRAMES - i] = self.data_x[idx - i]
                y[N_FRAMES - i] = self.data_y[idx - i]
            if idx + i < len(self.data_x):
                x[N_FRAMES + i] = self.data_x[idx + i]
                y[N_FRAMES + i] = self.data_y[idx + i]
        return preprocess_image(x.copy()), preprocess_image(y.copy())

class DataSetTrainTemporal(torch.utils.data.Dataset):
    def __init__(self, dir_x, dir_y):
        print(f'Loading temporal train dataset from {dir_x} and {dir_y}...')
        self.data_x = make_dataset(dir_x, True)
        self.data_y = make_dataset(dir_y, False)
        assert self.data_x.shape[-2:0] == self.data_y.shape[-2:0], "Data set size mismatch!"
        assert len(self.data_x) == len(self.data_y), "Data set length mismatch!"
        self.input_channels = self.data_x.shape[-3]

    def __len__(self):
        return 4 * len(self.data_x) # ~4 random crops per image per epoch

    def __getitem__(self, idx):
        x = np.zeros((2*N_FRAMES+1, self.data_x.shape[-3], TILE_SIZE, TILE_SIZE), dtype=self.data_x.dtype)
        y = np.zeros((2*N_FRAMES+1, self.data_y.shape[-3], TILE_SIZE, TILE_SIZE), dtype=self.data_y.dtype)
        f = np.random.randint(0, len(self.data_x))
        from_x = np.random.randint(0, self.data_x.shape[-1] - TILE_SIZE)
        from_y = np.random.randint(0, self.data_x.shape[-2] - TILE_SIZE)
        x[N_FRAMES] = self.data_x[f, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
        y[N_FRAMES] = self.data_y[f, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
        for i in range(1, N_FRAMES+1):
            if f - i >= 0:
                x[N_FRAMES - i] = self.data_x[f - i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
                y[N_FRAMES - i] = self.data_y[f - i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
            if f + i < len(self.data_x):
                x[N_FRAMES + i] = self.data_x[f + i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
                y[N_FRAMES + i] = self.data_y[f + i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
        x, y = random_flip(x, y)
        return preprocess_image(x.copy()), preprocess_image(y.copy())

# -----------------------------------------------------------
# reprojected image sequence data sets

class DataSetPredictReproj(torch.utils.data.Dataset):
    def __init__(self, dir_x):
        print(f'Loading reproj predict dataset from {dir_x}...')
        self.data_x = make_dataset(dir_x, True)
        self.data_p = make_dataset_pos(dir_x, self.data_x.shape, key='.dat')
        self.proj = load_matrices(dir_x, self.data_x.shape)
        self.input_channels = self.data_x.shape[-3]

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = np.zeros((2*N_FRAMES+1, self.input_channels, self.data_x.shape[-2], self.data_x.shape[-1]), dtype=self.data_x.dtype)
        x[N_FRAMES] = self.data_x[idx]
        for i in range(1, N_FRAMES+1):
            if idx - i >= 0:
                x[N_FRAMES - i] = reproject(self.data_x[idx - i], self.data_p[idx - i], self.proj[idx])
            if idx + i < len(self.data_x):
                x[N_FRAMES + i] = reproject(self.data_x[idx + i], self.data_p[idx + i], self.proj[idx])
        return preprocess_image(x)

class DataSetTestReproj(torch.utils.data.Dataset):
    def __init__(self, dir_x, dir_y):
        print(f'Loading reproj test dataset from {dir_x} and {dir_y}...')
        self.data_x = make_dataset(dir_x, True)
        self.data_y = make_dataset(dir_y, False)
        self.data_p = make_dataset_pos(dir_x, self.data_x.shape, key='.dat')
        self.proj = load_matrices(dir_x, self.data_x.shape)
        assert self.data_x.shape[-2:0] == self.data_y.shape[-2:0], "Data set size mismatch!"
        assert len(self.data_x) == len(self.data_y) == len(self.proj), "Data set length mismatch!"
        self.input_channels = self.data_x.shape[-3]

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = np.zeros((2*N_FRAMES+1, self.input_channels, self.data_x.shape[-2], self.data_x.shape[-1]), dtype=self.data_x.dtype)
        y = np.empty((self.data_y.shape[-3], self.data_y.shape[-2], self.data_y.shape[-1]), dtype=self.data_y.dtype)
        x[N_FRAMES], y = self.data_x[idx], self.data_y[idx]
        for i in range(1, N_FRAMES+1):
            if idx - i >= 0:
                x[N_FRAMES - i] = reproject(self.data_x[idx - i], self.data_p[idx - i], self.proj[idx])
            if idx + i < len(self.data_x):
                x[N_FRAMES + i] = reproject(self.data_x[idx + i], self.data_p[idx + i], self.proj[idx])
        return preprocess_image(x), preprocess_image(y)

class DataSetTrainReproj(torch.utils.data.Dataset):
    def __init__(self, dir_x, dir_y):
        print(f'Loading reproj train dataset from {dir_x} and {dir_y}...')
        self.data_x = make_dataset(dir_x, True)
        self.data_y = make_dataset(dir_y, False)
        self.data_p = make_dataset_pos(dir_x, self.data_x.shape, key='.dat')
        self.proj = load_matrices(dir_x, self.data_x.shape)
        assert self.data_x.shape[-2:0] == self.data_y.shape[-2:0], "Data set size mismatch!"
        assert len(self.data_x) == len(self.data_y) == len(self.data_p) == len(self.proj), "Data set length mismatch!"
        self.input_channels = self.data_x.shape[-3]

    def __len__(self):
        return 4 * len(self.data_x) # ~4 random crops per image per epoch

    def __getitem__(self, idx):
        x = np.zeros((2*N_FRAMES+1, self.input_channels, TILE_SIZE, TILE_SIZE), dtype=self.data_x.dtype)
        y = np.empty((self.data_y.shape[-3], TILE_SIZE, TILE_SIZE), dtype=self.data_y.dtype)
        f = np.random.randint(0, len(self.data_x))
        from_x = np.random.randint(0, self.data_x.shape[-1] - TILE_SIZE)
        from_y = np.random.randint(0, self.data_x.shape[-2] - TILE_SIZE)
        x[N_FRAMES] = self.data_x[f, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
        y = self.data_y[f, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE]
        for i in range(1, N_FRAMES+1):
            if f - i >= 0:
                x[N_FRAMES - i] = reproject(self.data_x[f - i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE], self.data_p[f - i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE], self.proj[f], (from_x, from_y))
            if f + i < len(self.data_x):
                x[N_FRAMES + i] = reproject(self.data_x[f + i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE], self.data_p[f + i, :, from_y:from_y+TILE_SIZE, from_x:from_x+TILE_SIZE], self.proj[f], (from_x, from_y))
        x, y = random_flip(x, y)
        return preprocess_image(x.copy()), preprocess_image(y.copy())

# -----------------------------------------------------------
# reprojection utility functions

def parse_proj(floats, shape):
    model = np.reshape(np.array(floats[0:16], dtype='float32'), (4, 4))
    view = np.reshape(np.array(floats[16:32], dtype='float32'), (4, 4))
    proj = np.reshape(np.array(floats[32:48], dtype='float32'), (4, 4))
    scale1 = np.array([[0.5, 0, 0, 0], [0, -0.5, 0, 0], [0, 0, 1, 0], [0.5, 0.5, 0, 1]], dtype='float32')
    scale2 = np.array([[shape[-1], 0, 0, 0], [0, shape[-2], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='float32')
    return model @ view @ proj @ scale1 @ scale2

def reproject(img, pos, MVPS, offset=(0, 0)):
    """ reproject image or tile into MVPS, where img[..., 3:6] are the volume texture coordinates """
    # prepare (model) position buffer
    tmp = np.ones((img.shape[-1], img.shape[-2], 4), dtype='float32')
    tmp[..., 0:3] = np.swapaxes(pos,  0, 2) * 2 - 1 # tc to model space and chw to whc
    # reproject, dehomogenize and round to pixel
    tmp = tmp @ MVPS
    tmp /= np.expand_dims(tmp[..., 3], axis=-1)
    px = np.round(tmp[..., 0:2]).astype('int')
    # translate to tile and clip
    px[..., 0] = np.clip(px[..., 0] - offset[0], 0, img.shape[-1] - 1)
    px[..., 1] = np.clip(px[..., 1] - offset[1], 0, img.shape[-2] - 1)
    # gather result
    px = np.swapaxes(px, 0, 2)
    result = np.zeros_like(img)
    result[:, px[1, ...], px[0, ...]] = img[:, ...]
    return result

def reproject_torch(img, pos, MVPS, offset=(0, 0)):
    with torch.no_grad():
        # prepare (model) position buffer
        tmp = torch.ones((img.size(-1), img.size(-2), 4), dtype=torch.float32)
        if img.is_cuda: tmp = tmp.cuda()
        tmp[..., 0:3] = pos.transpose(0, 2) * 2 - 1 # tc to model space and chw to whc
        # reproject, dehomogenize and round to pixel
        tmp = torch.matmul(tmp, MVPS)
        tmp /= torch.unsqueeze(tmp[..., 3], dim=-1)
        px = torch.round(tmp[..., 0:2]).to(torch.long)
        # translate to tile and clip
        px[..., 0] = torch.clamp(px[..., 0] - offset[0], 0, img.size(-1) - 1)
        px[..., 1] = torch.clamp(px[..., 1] - offset[1], 0, img.size(-2) - 1)
        # gather result
        px = torch.transpose(px, 0, 2)
        result = torch.zeros_like(img)
        result[:, px[1, ...], px[0, ...]] = img[:, ...]
    return result
