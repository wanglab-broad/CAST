# Installation Guide
## Development
Users should install the following packages prior to use the CAST package.

The environment requires 2-5 minutes to deploy acorrding to the Internet speed.

If users are using the Windows system, please directly use the `Anaconda powershell` to run following code.

1. Create a conda environment running python 3.9 (or above)

For example, create a new environment named `cast_demo` and activate it.
```
conda create -y -n cast_demo python=3.9
conda activate cast_demo
```
2. Run the following script to deploy the environment. Users should change to the CAST package folder `cd $CAST_Package_PATH`.
```
#### If CPU only ###
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
conda install -c dglteam dgl
pip install -r ./requirements.txt
#### If GPU available ###
conda install -y -c pytorch pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3
conda install -y -c dglteam dgl-cuda11.3==0.9.1
pip install -r ./requirements.txt
```


## Packages dependencies
The versions of `pytorch` and `dgl` are listed above. Other packages are listed in the `requirements.txt`.


# Demo
We provide with several demos to demonstrate the functions in CAST package.

Users can use following code to open the `Jupyter notebook` (We recommend to use `Chrome` to open the jupyter notebook).
```
cd $CAST_Package_PATH/demo
jupyter notebook
#### If remote kernel ####
jupyter notebook --ip=0.0.0.0 --port=8800
#### If dead kernel ####
jupyter notebook --NotebookApp.max_buffer_size=21474836480
```
## Demo 1 CAST_Mark
In this demo, CAST_Mark can captures the common spatial features across multiple samples

Users should first replace `$CAST_package_path` with the CAST package Path in the first cell.

## Demo 2 CAST_Stack_Align_S4_to_S1
In this demo, Stack_Align can align two samples together.

Users should first replace `$CAST_package_path` with the CAST package Path in the first cell.

## Demo 3 CAST_project
In this demo, CAST_projectiong will project one sample to another one.

Users should first replace `$CAST_package_path` with the CAST package Path in the first cell.
