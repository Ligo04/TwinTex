# Multi-Mask RePaint (MMRP)

### Pretrained Models Usage

- Download the pretrained MMRP model from [here](https://www.dropbox.com/scl/fi/mc3gsvsbxhp27sl0rknbw/ema_0.9999_151161.pt?rlkey=te3n8gxt3op0zkpxltxxnba79&dl=0) and copy it to `./ata/pretrained/*`.


## Structure of the document and part of the code

- `extract_mask.py`：Image Preprocessing.
- `mmrp_inpainting.py`： MMRP main.
- `inpaint.py`: Repainted .
- `conf_mgt/*`：Parameter file handling, file writing.
- `confs/*`：Parameter file directory.
- `data/datasets/*`：Ground truth and mask.
- `data/pretrained/*`：Pre-trained models.
- `guided_diffusion/*`：Diffusion model related.
- `input/*`：Catalog of input image .
- `log/*`：Catalog of intermediate results.
- `result/*`：Catalog of final results.
- `temp`：Directory of temporary documents (conversions, cuts, etc).

## Houdini python Environment Installation

For Windows, use the following commands to install Pip ([get-pip.py](https://bootstrap.pypa.io/get-pip.py)) and external python modules in Houdini. (Houdini python in `./Houdini 19.5.xx/python39/`)

```
# install pip
python3.9.exe get-pip.py
# requirements
python3.9.exe -m pip install -r requirements.txt
# pytroch
python3.9.exe -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Others

Temp results are added to the `Results/inpainting` directory:

```bash
├── ./data/datasets/gt_keep_masks
│   ├── planexxxx_twintex46365/000000.png  # Extracted masks identifying missing regions, black is missing,white is known
│   ├── planexxxx_twintex46365_filter_mask/000000.png  # Masks for areas that don't need to be patched up
│   ├── planexxxx_twintex46365_sketch1/000000.png # Mask chunk 1
│   ├── planexxxx_twintex46365_sketch2/000000.png # Mask chunk 2
│   ├── planexxxx_twintex46365_sketch3/000000.png # Mask chunk 3

├── ./data/datasets/gts
│   ├── plane298/000000.png  # ground truth,512x512
./confs/
│   ├── plane298_twintex46365.yml
```



