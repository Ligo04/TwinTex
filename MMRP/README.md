# Multi-Mask RePaint (MMRP)

The core idea is to utilize the high sensitivity of the diffusion model to the initial input noise during the sampling phase in order to indirectly achieve suppression of the overfitting effect.
There are two key improvements:

- The missing regions are chunked and each initialized to the same Gaussian noise
- Initialize different regions of the chunk with Gaussian noise of different parameters

### Pretrained Models Usage

- Download pretrained model from [here](https://www.dropbox.com/scl/fi/mc3gsvsbxhp27sl0rknbw/ema_0.9999_151161.pt?rlkey=te3n8gxt3op0zkpxltxxnba79&dl=0).

- And then place in `./data/pretrained/*`


## Structure of the document and part of the code

- `extract_mask.py`：Image Preprocessing
- `mmrp_inpainting.py`： MMRP main
- `inpaint.py`: repainted 
- `conf_mgt/*`：Parameter file handling, file writing
- `confs/*`：Parameter file directory
- `data/datasets/*`：ground truth and mask
- `data/pretrained/*`：Pre-trained models
- `guided_diffusion/*`：Diffusion model related
- `input/*`：Catalog of input image 
- `log/*`：Catalog of intermediate results
- `result/*`：Catalog of final results
- `temp`：Directory of temporary documents (conversions, cuts, etc.)

## houdini python Environment Installation

1. Windows follows the path to navigate to the Houdini installation directory and then to the Python folder (e.g. python39). If pip.py does not exist, you will need to download [pip](https://bootstrap.pypa.io/get-pip.py) first, then open a Windows terminal in this folder and enter the command to install pip.

   ```
   python3.9.exe get-pip.py
   ```

   Place the project's `requirements.txt` in the corresponding python directory, and install the environment via `requirements.txt`.

   ```
   python3.9.exe -m pip install -r requirements.txt
   ```

   `pytorch` install:

   ```
   python3.9.exe -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

Some temp results is added to the `Results/inpainting` directory:

```bash
├── ./data/datasets/gt_keep_masks
│   ├── planexxxx_twintex46365/000000.png  # Extracted masks identifying missing regions, black is missing, white is known
│   ├── planexxxx_twintex46365_filter_mask/000000.png  # Masks for areas that don't need to be patched up
│   ├── planexxxx_twintex46365_sketch1/000000.png # Mask chunk 1
│   ├── planexxxx_twintex46365_sketch2/000000.png # Mask chunk 2
│   ├── planexxxx_twintex46365_sketch3/000000.png # Mask chunk 3

├── ./data/datasets/gts
│   ├── plane298/000000.png  # ground truth，512x512
./confs/
│   ├── plane298_twintex46365.yml
```



