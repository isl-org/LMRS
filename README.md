# LMRS: Learned Manifold Random Search

This code repository includes the source code for the [Paper](https://arxiv.org/abs/XXX.XXXXX):

```
Learning to Guide Random Search
Ozan Sener, Vladlen Koltun
International Conference on Learning Representations (ICLR) 2020 
```

The experimentation framework is based on [Ray](https://github.com/ray-project/ray) and extends the implementation of [ARS](https://github.com/modestyachts/ARS).

The source code is released under the MIT License. See the License file for details.

Please note that this is the minimal implementation of the LMRS for MuJoCo, we will update the repo with the additional code for XFoil, Pagmo, and synthetic experiments. 


# Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch>1.0, click, numpy, torchvision, tqdm, scipy, Pillow, ray``

The code is only tested in ``Python 3`` using ``Anaconda`` environment.

If you want to run the MuJoCo experiments, install OpenAI Gym (version 0.9.3) and MuJoCo(version 0.5.7) following the [instructions](https://github.com/openai/gym).

If you want to run the AirFoil experiments, install [XFoil](https://web.mit.edu/drela/Public/web/xfoil/) and make sure the binary is in the `$PATH`.

If you want to run the continous optimization benchmark, install ``Pagmo`` following [esa/pagmo2](https://github.com/esa/pagmo2).

# Usage
Experiment specific parameters are provided as a json file. See the `hc.json` for an example.

To run an example experiment, use the command: 
```bash
python mujoco_experiments.py --param_file=./hc.json
```

# Contact
For any question, you can contact ozan.sener@intel.com

# Citation
If you use this codebase or any part of it for a publication, please cite:
```
@inproceedings{ICLR2020_Sener_Koltun,
title={Learning to Guide Random Search},
author={Ozan Sener and Vladlen Koltun},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=B1gHokBKwS}
}
```
