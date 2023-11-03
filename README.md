Lightweight Face Recognition via Shrinking Teacher-Student Networks
=================================================
This repository contains an implementation of the distillation method mentioned in this [paper](https://arxiv.org/abs/1905.10620). Using the code from this repository, you can train a lightweight network to recognize faces for embedded devices.

<br>

# Data preparation

  1) Download dataset from [here](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view).
  2) Extract images using [this](https://github.com/david-svitov/margindistillation/blob/master/data_prepare/bin_get_images.ipynb).
  3) Get the labels for each image and create annotations.csv using: create_annotation.ipynb
  4) Or get download dataset from [here](https://iitbhuacin-my.sharepoint.com/:u:/g/personal/rakshit_sawhney_ece21_iitbhu_ac_in/ERjGwLCISttBvIfvunv4oIkB_ZoRw5WkbB8smZguNc6nyQ?e=3Bylo8) to skip above 2 steps.
  5) Create a folder in home directory by the name of CASIA_dataset
  6) Save the Images Folder and annotations.csv in it

<br>

# Training
Create folder for saving model:
```
mkdir output_logs
```

Start training using:
```
python3 run.py
```


## Additional flags
|Flag             | Default Value        |Description
|:----------------|:--------------------:|:----------------------------------------------
|batchsize        |  8                   | batch size
|epochs           |  10                  | number of epochs to train
|lr               |  0.1                 | learning rate
|lmda             |  0.25                | lambda
|device           |  cpu                 | device using (eg. cuda , mps)
|workers          |  0                   | number of workers


<br>


# Using HPC
Paramshivay, Supercomputer of IIT BHU (Varanasi) was used for training of this model. 

### Setting Up Anaconda
Install miniconda locally for more reliable experience and easy package managing.

Download Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Install Miniconda
```
bash Miniconda3-latest-Linux-x86_64.sh
```
Activate Miniconda
```
source miniconda3/bin/activate
```

### Using Interactive Terminal
To get a node with 1 gpu and 20 cores for 1 hour for running the code live.<br>
(Interactive session will over as soon as time is over or your terminal is disconnected from HPC and execution will be interupted.)

```
srun --partition=gpu --nodes=1 --gres=gpu:1 --time=01:00:00 --ntasks-per-node=20  --pty bash -i
```

### Using Shell Script to Submit the Job
Submit the Job to SLURM for completion.<br>
(No need to be connected to HPC for file to keep executing)
```
sbatch script.sh
```

## Contributors
- [Raunak Pandey](https://github.com/patrick-batman)
- [Rakshit Sawhney](https://github.com/RakshitSawhney)

