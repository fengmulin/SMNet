# SMNet

Real-Time Text Detection in Traffic, Industrial, and  Scenes with Similar Mask





## Environment
The environment and usage are based on: [DBNet](https://github.com/MhLiao/DB)
```bash
conda create -n SM python==3.7
conda activate SM


pip install -r requirement.txt

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


git clone https://github.com/MhLiao/SMNet.git
cd SMNet/

echo $CUDA_HOME
cd assets/ops/dcn/
python setup.py build_ext --inplace

```

## Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4
```

## Testing
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.65
```
## Dataset
[MBTSC](链接：https://pan.baidu.com/s/1rvOI3OsQJfDjM8-abPZENg 
提取码：bsv1)

## Acknowledgement
Thanks to [DBNet](https://github.com/MhLiao/DB) for a standardized training and inference framework. 


