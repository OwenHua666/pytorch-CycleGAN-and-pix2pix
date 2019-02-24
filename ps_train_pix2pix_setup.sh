pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
python train.py --dataroot /dataset/holopix --name holopix-pix2pix-no-l1-upsampling-10k-batch-1-date-02-24-001 \
--model pix2pix --direction AtoB --gpu_ids 0,1,2,3 --batch_size 4 \
--checkpoints_dir /persistent --output_nc 1 --dataset_mode holopix \
--dataset_num 10000 --preprocess resize --load_size 256 --ignore_l1 1
