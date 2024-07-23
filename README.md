# SMNet

Real-Time Text Detection in Traffic, Industrial, and  Scenes with Similar Mask





## Environment
The environment and usage are based on: [DBNet](https://github.com/MhLiao/DB)

'conda create -n SM python==3.7
conda activate SM


pip install -r requirement.txt

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


git clone https://github.com/MhLiao/SMNet.git
cd SMNet/

echo $CUDA_HOME
cd assets/ops/dcn/
python setup.py build_ext --inplace'


## Dataset
[MBTSC](链接：https://pan.baidu.com/s/1rvOI3OsQJfDjM8-abPZENg 
提取码：bsv1)

## Acknowledgement
Thanks to [DBNet](https://github.com/MhLiao/DB) for a standardized training and inference framework. 


