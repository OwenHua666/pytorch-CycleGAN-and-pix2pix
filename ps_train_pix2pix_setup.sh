pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
python train.py --dataroot /dataset/holopix --name holopix-pix2pix-1k-batch-3-date-02-21-002 \
--model pix2pix --direction AtoB --gpu_ids 0,1,2 --batch_size 3 \
--checkpoints_dir /persistent --output_nc 1 --dataset_mode holopix \
--preprocess resize --load_size 256 --dataset_num 1000
