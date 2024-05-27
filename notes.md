### details

python 3.11.8


### venv

python -m venv mvsnerf-venv
source mvsnerf-venv/bin/activate


### requirements

pip3 install torch torchvision torchaudio
pip3 install lightning imageio pillow scikit-image opencv-python configargparse lpips kornia warmup_scheduler matplotlib test-tube imageio-ffmpeg rich

git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python3 setup.py install


CUDA_VISIBLE_DEVICES=0  python train_mvs_nerf_pl.py \
    --imgScale_train 0.5 --imgScale_test 0.5 --pad 12 \
    --expname mvs-nerf-is-all-your-need \
    --num_epochs 6 --N_samples 128 --use_viewdirs --batch_size 1 \
    --dataset_name dtu \
    --datadir /home/r2tp/Repos/mvsnerf/train_data/dtu_example \
    --N_vis 6

CUDA_VISIBLE_DEVICES=0  python train_mvs_nerf_finetuning_pl.py  \
    --dataset_name dtu_ft --datadir /home/r2tp/Repos/mvsnerf/train_data/dtu_example \
    --expname scan1-ft  --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0   --pad 24 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1


### resources

https://www.dataquest.io/blog/a-complete-guide-to-python-virtual-environments/
https://lightning.ai/docs/torchmetrics/stable/
https://lightning.ai/docs/pytorch/stable/upgrade/from_1_4.html


### requirements

see requirements.txt

-ffmpeg

pip install inplace-abn

CUDA_VISIBLE_DEVICES=0  python train_mvs_nerf_pl.py \
   --expname test_run \
   --num_epochs 6 \
   --use_viewdirs \
   --dataset_name dtu \
   --datadir /home/r2tp/Repos/mvsnerf-data/dtu_example/dtu