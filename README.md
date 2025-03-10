# TS-LIF
[ICLR 2025] Official Implementation of "TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting"

## Related Papers
* TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting, [ICLR 2025], (https://arxiv.org/abs/2503.05108)
* Efficient and Effective Time-Series Forecasting with Spiking Neural Networks, [ICML 2024], (https://arxiv.org/pdf/2402.01533).


## Installation
To install TS-LIF within (SeqSNN) in a new conda environment:
```
conda create -n SeqSNN python=[3.8, 3.9, 3.10]
conda activate SeqSNN
git clone https://github.com/microsoft/SeqSNN/
cd SeqSNN
pip install .
```

If you would like to make changes and run your experiments, use:

`pip install -e .`

## Training
Take the `TS-GRU` model as an example:

`python entry.tsforecast.py /path/exp/forecast/spikegru/spikegru_electricity.yml"`

You can change the `yml` configuration files as you want.

## Datasets

Metr-la and Pems-bay datasets are available on [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) and [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g).  
Solar and Electricity datasets can be downloaded from [GitHub](https://github.com/laiguokun/multivariate-time-series-data).

## Acknowledgement
This repo is built upon (Lv's Repo) [https://github.com/microsoft/SeqSNN], and we show sincerely thanks for @Changze Lv's initial contribution. 

## Citing

If our work has inspired your research, you can  consider citing us:
```BibTeX
@inproceedings{shibots,
  title={TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting},
  author={SHIBO, FENG and Feng, Wanjin and Gao, Xingyu and Zhao, Peilin and Shen, Zhiqi},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

@article{lv2024efficient,
  title={Efficient and effective time-series forecasting with spiking neural networks},
  author={Lv, Changze and Wang, Yansen and Han, Dongqi and Zheng, Xiaoqing and Huang, Xuanjing and Li, Dongsheng},
  journal={arXiv preprint arXiv:2402.01533},
  year={2024}
}
