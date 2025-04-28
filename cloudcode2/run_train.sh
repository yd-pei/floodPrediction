#!/bin/bash
conda init bash
conda activate flood
# python ./ondevice_training.py
# python ./t12_100_13.py
python ./cloud_train1.py
python ./cloud_train2.py
# python ./t24_200_34.py