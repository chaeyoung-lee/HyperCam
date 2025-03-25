# HyperCam
Low-Power Onboard Computer Vision for IoT Cameras ([paper](https://arxiv.org/abs/2501.10547))

Chae Young Lee, Pu (Luke)Yi, Maxwell Fite, Tejus Rao, Sara Achour, Zerina Kapetanovic

Stanford University

### Usage

```shell
pip install -r requirements.txt
python test.py -c config/bloomfilter.yaml
```

Edit the configuration file to choose your choice of encoder and dataset.
* Encoder: `naive`, `hypercam-countsketch`, `hypercam-bloomfilter`
* Dataset: `mnist`, `fashion-mnist`

### Citation
If you use this code, please cite the paper as following.
```
@article{hypercam25,
  title={HyperCam: Low-power Onboard Computer Vision for IoT Cameras},
  author={Chae Young Lee, Pu (Luke) Yi, Maxwell Fite, Tejus Rao, Sara Achour, Zerina Kapetanovic},
  year={2024}
}
```
