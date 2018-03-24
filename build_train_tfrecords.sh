#!/usr/bin/env


python3 build_tfrecords.py --HR_dir dataset/2018/train/DIV2K_train_HR \
                           --LR_dir dataset/2018/train/DIV2K_train_LR_x8 \
                           --output_file dataset/2018/train_2018_bicubic_X8.tfrecords

python3 build_tfrecords.py --HR_dir dataset/2018/train/DIV2K_train_HR \
                           --LR_dir dataset/2018/train/DIV2K_train_LR_mild \
                           --output_file dataset/2018/train_2018_unknown_mild.tfrecords

python3 build_tfrecords.py --HR_dir dataset/2018/train/DIV2K_train_HR \
                           --LR_dir dataset/2018/train/DIV2K_train_LR_difficult \
                           --output_file dataset/2018/train_2018_unknown_difficult.tfrecords

python3 build_tfrecords.py --HR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_mild/HR \
                           --LR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_mild/LR \
                           --output_file /mnt/nas/Dataset/NTIRE2018/aligned/2018_mild/train_2018_unknown_mild.tfrecords

python3 build_tfrecords.py --HR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_difficult/HR \
                           --LR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_difficult/LR \
                           --output_file /mnt/nas/Dataset/NTIRE2018/aligned/2018_difficult/train_2018_unknown_difficult.tfrecords

python3 build_tfrecords.py --HR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_1/HR \
                           --LR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_1/LR \
                           --output_file /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_1/train_2018_unknown_wild_1.tfrecords

python3 build_tfrecords.py --HR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_2/HR \
                           --LR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_2/LR \
                           --output_file /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_2/train_2018_unknown_wild_2.tfrecords

python3 build_tfrecords.py --HR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_3/HR \
                           --LR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_3/LR \
                           --output_file /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_3/train_2018_unknown_wild_3.tfrecords

python3 build_tfrecords.py --HR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_4/HR \
                           --LR_dir /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_4/LR \
                           --output_file /mnt/nas/Dataset/NTIRE2018/aligned/2018_wild_4/train_2018_unknown_wild_4.tfrecords
