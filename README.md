# Neural Denoising for Path Tracing of Medical Volumetric Data

Source code for the paper "Neural Denoising for Path Tracing of Medical Volumetric Data"

## Build

Setup virtual python environment:

    virtualenv -p python3 env
    source env/bin/activate
    pip3 install -r requirements.txt
 
## Train:

Single autoencoder architecture:

    python src/train.py /path/to/data/noisy /path/to/data/clean /path/to/data/validation/noisy /path/to/data/validation/clean
    
Dual autoencoder architecture:

    python src/train.py /path/to/data/noisy /path/to/data/clean /path/to/data/validation/noisy /path/to/data/validation/clean -t dual
    
GAN-based dual autoencoder architecture:

    python src/train_relgan.py /path/to/data/noisy /path/to/data/clean /path/to/data/validation/noisy /path/to/data/validation/clean --big
    
Dual autoencoder architecture with temporal reprojection:

    python src/train_reproj.py /path/to/data/noisy /path/to/data/clean /path/to/data/validation/noisy /path/to/data/validation/clean --feature_select

Additional command line arguments are listed in the beginning of each source file.
  
## Denoise:

    python src/denoise.py /path/to/model.pt /path/to/data/noisy /path/to/data/clean
  
Additional command line arguments are listed in the beginning of the source file.
  
## Pretrained models

Pretrained models for all configurations are available at models/*.pt.
To load a model into pytorch or access to the weights:

    cd src/
    python3
    >>> import torch
    >>> import models
    >>> torch.load('../models/single_color_only.pt').state_dict()
  
## Data Layout

For each stored frame under /path/to/data/noisy the following data is expected:
  - \<Frame\>.hdr: input color data
  - \<Frame\>_pos.hdr: primary scatter positions
  - \<Frame\>_norm1.hdr: primary normals
  - \<Frame\>_norm2.hdr: color after first surface approximation
  - \<Frame\>_alb1.hdr: primary albedo
  - \<Frame\>_alb2.hdr: secondary albedo
  - \<Frame\>_vol1.hdr: primary volumetrics
  - \<Frame\>_vol2.hdr: secondary volumetrics
  
For each stored frame under /path/to/data/clean the following data is expected:
  - \<Frame\>.hdr: target color data
  
  
For temporal reprojection, each noisy frame is additionally expected to provide:
  - \<Frame\>_pos.dat: binary high-precision primary scatter positions
  - \<Frame\>_matrices.txt: three 4x4 floating point matrices (model, view, proj), i.e. 12 rows with 4 IEE plaintext floats each

File names are recursively globbed and then sorted alphanumerically, thus file names do not need to match exactly between noisy and clean data, just required to sort into the same order.
