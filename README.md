# HyperCam
Low-Power Onboard Computer Vision for IoT Cameras

### Usage

Dockerfile is provided for usage on Docker, but it is not necessary for reproducing results. Using Python 3, run

```shell
pip install -r requirements.txt
```

To evaluate all classifiers on all available datasets, run

```shell
python test_all.py
```

For single classifier usage, run

```shell
python test.py -c config/bloomfilter.yaml
```

Edit the configuration file to choose your choice of encoder and dataset.
* Encoder: `naive`, `hypercam-countsketch`, `hypercam-bloomfilter`
* Dataset: `mnist`, `fashion-mnist`
