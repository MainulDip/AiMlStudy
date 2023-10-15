### Env:
- Conda : Package Manager
- Anaconda : Software Distribution. Anaconda is larger and comes with a vast array of pre-installed packages
- Mini Conda : Software Distribution, timed down version of the Anaconda. MiniConda only includes Conda and Python.

### MiniConda Installation Linux:
- Download the installer (Miniconda3-latest-Linux-x86_64.sh).
- run "bash Miniconda3-latest-Linux-x86_64.sh" then press "enter" to enter into license agreement, then press "q" and enter "Yes" to accept.
- Check : run "conda list" to check the installation by reopening the terminal

### Conda Config:
- Windows Env Var: run `where conda` in anaconda/miniconda terminal and add those path to the system environmental variable path
- run `conda init bash` or `conda init <shell-interface>` to run `conda activate env-name`
### New Projects:
```sh
# cd into project dir and run
conda create --prefix ./env pandas numpy matplotlib scikit-learn jupyter
# then activate the environment as printed in the terminal or 
# from the project root run this to separate the project file form env file, run pwd for current folder location
conda activate ./env

# Tensorflow (GPU) in conda environment
# Install cuda toolkit
# Install cuda inside conda env with nvidia channel
conda install cuda -c nvidia
conda create -p ./env tensorflow-gpu pandas numpy matplotlib scikit-learn jupyter 

# install jupyter notebook if not before
conda install jupyter
conda install tensorflow-gpu

# run jupyter notebook cmd from the root of the project to start
jupyter notebook

# deactivate conda env
conda deactivate

# list of env
conda env list

# exit form jupyter server
ctrl+c

# Store conda env on a yml file to use on another machine (activate the environment first)
conda env export > env.yml

# exporting environment yml file as cross-platform compatibility only with core packages (change the name and delete prefix from the exported yml file )
conda env export --ignore-channels --from-history > shared-env-file.yml

# installing conda env form a file name env.yml
conda env create -f shared-env-file.yml -p ./target-directory-path

# check version and upgrade packages (activate the environment first)
conda list
conda list <package-name>
conda search <package-name> # check the suggested update version
conda update <package-name>
conda uninstall <package-name> <package-name> <package-name>
```

### Tensorflow in Anaconda/Miniconda Jupyter (Only Nvidia's CUDA Enabled GPUs):
* Note: Tensorflow latest release (GPU) is only officially compatible with Ubuntu or Via `Windows WSL2`. Follow official Doc for installation
* Install the exact version of GPU packages, follow https://www.tensorflow.org/install/pip#windows-wsl2_1
* https://docs.nvidia.com/cuda/wsl-user-guide/index.html
* https://learn.microsoft.com/en-us/windows/wsl/basic-commands

First Part : Host Setup ----------------------------------
- install latest graphics driver on host machine

Second Part : WSL Setup: ---------------------------------

After installing, within wls2, move forward with Conda environment
1. Install Tensorflow Specified Version of CUDA Tool kit
2. Install Tensorflow Specified Version of CUDNN (Cuda Neural Network)
3. Install Matching TensorRT against `Cuda Toolkit` From Nvidia

Third Part : Conda Setup on WSL
1. Create empty conda env with specified python version `conda create -p ./env python=3.9`
2. Install pip through conda to use pip inside conda environment `conda install pip`
3. Now install all Tensorflow and other packages as docs suggest
4. Install Jupyter Notebook `pip install notebook` and run using `jupyter notebook`

### Frequently Used WSL Commands:
- WSL list installed distributions and log into : `wsl --list` and `wsl -d Ubuntu -u username` 
- WSL Shutdown : wsl --shutdown
- delete distro Ubuntu `wsl --unregister <distroName> where <distroName>`

### Verify If Tensorflow is using GPU :
```python
# Import Tensorflow
import tensorflow as tf
# import tensorflow_hub as hub
print("TF version:", tf.__version__)
# print("TF Hub version:", hub.__version__)

# Check For GPU availability (o use GPU change Runtime to GPU)
print("GPU", "available (Yess!!!!!!!!)" if tf.config.list_physical_devices("GPU") else "not Available")
```
## Tensorflow GPU Docker Compose (Not Verified Locally Yet):
```yml
#version: "3.3"

services:
  jupyter:  # you can change this to whatever you want.
    container_name: computer-vison
    image: tensorflow/tensorflow:2.2.2-gpu-py3-jupyter
    volumes:
      - "./:/tf/notebooks"
    ports:
     - "8888:8888"
    deploy:
      resources:
        reservations:
          devices:
          -  driver: nvidia
             count: all
             capabilities: [gpu]
```
### Jupyter Notebook:
- jupyter notebook's code runs on execution (run) sequence and the number before the cell indicate the order of execution.
- shift + enter to run a specific cell
- enter into command mode : esc
    - from cmd mode switch to markdown mode : m
    - from cmd mode switch to code : y
- intelligence : tab
- new cell above the current cell : form cmd mode (esc) press a
- new cell bellow : from cmd mode (esc) press b
- manually save : from cmd mode (esc) press s.
- delete cell : from cmd mode press dd.

jupyter notebook --no-browser --ip=192.168.0.12 --port=7000
### Jupyter Notebook Code Intellisense:
- tab : code suggestion
- shift + tab : docs
### Jupyter Notebook Bash/Sh commands:
- !<any command>
- !ls # will return list directory command
### Install conda package from Jupyter Notebook:
```py
# sys.prefix will return the current path
import sys
!conda install --yes --prefix {sys.prefix} seaborn
```
### Google Colab Workflow:
Google colab is a virtualized Jupyter Notebook Environment.
* Uploading File: Better to upload through the Google Drive Interface
* Mounting Drive: Mount the drive through Colab Interface 
* Unzipping.
```sh
# get the file location by copying the path 
# !unzip "files-path" -d "unzipping-destination"
!unzip "drive/MyDrive/Colab Notebooks/Data/dog-breed-identification.zip" -d "drive/MyDrive/Colab Notebooks/Data/model-dog-breed-identification/"
```