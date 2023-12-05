module load anaconda

conda create -n fyp_strats_37 python=3.7 jupyter nb_conda_kernels

source activate fyp_strats_37

conda install tensorflow=1.15.0=gpu_py37h0f0df58_0

conda install scikit-learn=0.24.2=py37ha9443f7_0

conda install pandas=1.3.4=py37h8c16a72_0

conda install tqdm=4.62.3=pyhd3eb1b0_1

conda install numpy=1.21.2=py37h20f2e39_0

conda install cudatoolkit=10.0.130=0

pip install h5py==2.10.0

pip install matplotlib
