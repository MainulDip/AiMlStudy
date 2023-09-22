### Env:
- Conda : Package Manager
- Anaconda : Software Distribution. Anaconda is larger and comes with a vast array of pre-installed packages
- Mini Conda : Software Distribution, timed down version of the Anaconda. MiniConda only includes Conda and Python.

### MiniConda Installation Linux:
- Download the installer (Miniconda3-latest-Linux-x86_64.sh)
- run "bash Miniconda3-latest-Linux-x86_64.sh" then press "enter" to enter into license agreement, then press "q" and enter "Yes" to accept.
- Check : run "conda list" to check the installation by reopening the terminal
### New Projects:
```sh
# cd into project dir and run
conda create --prefix ./env pandas numpy matplotlib scikit-learn jupyter
# then activate the environment as printed in the terminal or 
# from the project root run this to separate the project file form env file, run pwd for current folder location
conda activate ./env

# install jupyter notebook if not before
conda install jupyter

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

# installing conda env form a file name env.yml
conda env create -f=env.yml
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
- delete cell : from cmd mode press dd
