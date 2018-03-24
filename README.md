# TF_SRDense

Super-resolution based on SRDenseNet

### Test Enviroment
```angular2html
python 3.5
    - Numpy >= 1.14.0
    - Tensorflow >= 1.3.0
    - horovod
```

### Repository Architecture
```angular2html
- Data Pipeline
    - TFrecords
        build_tfrecords.py
        tfrecords_reader.py
        build_train_tfrecords.sh
      data_generator.py
- Training Pipeline
    multi_trainer.py
    config_training.py
- Testing Pipeline
    tester.py
    config_testing.py
- EDSR
    models/EDSR.py
    models/ops.py
- Utils
    config_helpers.py
```

### Usage
1. Copy or build TFrecords for training data
- Copy ready-made TFrecords for NTIRE2017 and NTIRE2018
- Or, make TFrecords from `.png` files (See `build_train_tfrecords.sh`)

2. Train the model

Modify `config_training.py` as you want.
Then, run `*_trainer*.py`

3. Test the model

Copy `.png` files from `nas/dataset/NTIRE*/`.

Modify `config_testing.py` as you want.
Then, run `*_tester.py`


## Multi GPU Training Example
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    python3 multi_trainer.py
```