# load-forecasting-resnet
This repository contains codes for the paper [Short-term Load Forecasting with Deep Residual Networks](https://ieeexplore.ieee.org/document/8372953).

## ISO-NE test case
We opensource in this repository the model used for the ISO-NE test case. Code for ResNetPlus model can be found in /ISO-NE/ResNetPlus_ISONE.py

The dataset contains load and temperature data from 2003 to 2014.

## North American test case
The code for the North American test case is added. Learning rate decay is added to produce more stable results.

## Updates
### (2021-05-11) Competition: [DAY-AHEAD ELECTRICITY DEMAND FORECASTING: POST-COVID PARADIGM](https://ieee-dataport.org/competitions/day-ahead-electricity-demand-forecasting-post-covid-paradigm)
The implementation of the proposed model in Pytorch ranked 5# (team 19) in this competition.

## Citation
If you find the codes useful in your research, please consider citing:

    @article{chen2018short,
      title={Short-term load forecasting with deep residual networks},
      author={Chen, Kunjin and Chen, Kunlong and Wang, Qin and He, Ziyu and Hu, Jun and He, Jinliang},
      journal={IEEE Transactions on Smart Grid},
      year={2018},
      publisher={IEEE}
    }

## License
[MIT LICENSE](LICENSE)
