# Third-party notices

Plumb is released under the MIT License (`LICENSE`). `src/harness/perception/` is adapted from the author's earlier
Plumb agent harness, under the same copyright and license. SAM 3 and Pi3 are linked as git submodules in
`third_party/` (their own repositories, pinned to the tested commits). They and the other software and models
below are not part of this repository's code and are used under their own licenses.

| Component | Upstream | License |
| --- | --- | --- |
| MuJoCo | https://github.com/google-deepmind/mujoco | Apache-2.0 |
| NumPy, SciPy | https://numpy.org/, https://scipy.org/ | BSD-3-Clause |
| OpenCV | https://opencv.org/ | Apache-2.0 |
| Pillow | https://python-pillow.org/ | MIT-CMU |
| Matplotlib | https://matplotlib.org/ | Matplotlib License |
| FFmpeg | https://ffmpeg.org/ | LGPL-2.1+ / GPL-2+ |
| SAM 3 (code and weights) | https://github.com/facebookresearch/sam3, https://huggingface.co/facebook/sam3 | SAM License |
| Pi3 (code) | https://github.com/yyfz/Pi3 | BSD-3-Clause |
| Pi3X (weights) | https://huggingface.co/yyfz233/Pi3X | CC BY-NC 4.0 (non-commercial only) |
| PyTorch, decord, safetensors, Hugging Face Hub | https://pytorch.org/, https://github.com/dmlc/decord, https://github.com/huggingface | BSD-3-Clause, Apache-2.0 |
| h5py, Requests (benchmark) | https://www.h5py.org/, https://requests.readthedocs.io/ | BSD-3-Clause, Apache-2.0 |

The benchmark (`benchmark/`) downloads its videos and ground truth from these datasets, which are not included:

| Dataset | Upstream | License |
| --- | --- | --- |
| PhysInOne | https://huggingface.co/datasets/vLAR/PhysInOne | CC BY-NC-SA 4.0 (non-commercial) |
| Physion v1.5 | https://physion-benchmark.github.io/physion_v15 | not stated by the release |

## Citations

```bibtex
@inproceedings{todorov2012mujoco,
  title     = {MuJoCo: A physics engine for model-based control},
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle = {2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages     = {5026--5033},
  year      = {2012}
}

@misc{carion2025sam3segmentconcepts,
  title         = {SAM 3: Segment Anything with Concepts},
  author        = {Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
  year          = {2025},
  eprint        = {2511.16719},
  archivePrefix = {arXiv}
}

@misc{zhou2026physinonevisualphysicslearning,
  title         = {PhysInOne: Visual Physics Learning and Reasoning in One Suite},
  author        = {Siyuan Zhou and Hejun Wang and Hu Cheng and Jinxi Li and Dongsheng Wang and Junwei Jiang and Yixiao Jin and Jiayue Huang and Shiwei Mao and Shangjia Liu and Yafei Yang and Hongkang Song and Shenxing Wei and Zihui Zhang and Peng Huang and Shijie Liu and Zhengli Hao and Hao Li and Yitian Li and Wenqi Zhou and Zhihan Zhao and Zongqi He and Hongtao Wen and Shouwang Huang and Peng Yun and Bowen Cheng and Pok Kazaf Fu and Wai Kit Lai and Jiahao Chen and Kaiyuan Wang and Zhixuan Sun and Ziqi Li and Haochen Hu and Di Zhang and Chun Ho Yuen and Bing Wang and Zhihua Wang and Chuhang Zou and Bo Yang},
  year          = {2026},
  eprint        = {2604.09415},
  archivePrefix = {arXiv}
}

@inproceedings{bear2021physion,
  title     = {Physion: Evaluating Physical Prediction from Vision in Humans and Machines},
  author    = {Bear, Daniel M. and Wang, Elias and Mrowca, Damian and Binder, Felix J. and Tung, Hsiao-Yu Fish and Pramod, R. T. and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and Fei-Fei, Li and Kanwisher, Nancy and Tenenbaum, Joshua B. and Yamins, Daniel L. K. and Fan, Judith E.},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year      = {2021}
}

@article{wang2025pi,
  title   = {$\pi^3$: Permutation-Equivariant Visual Geometry Learning},
  author  = {Wang, Yifan and Zhou, Jianjun and Zhu, Haoyi and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Pang, Jiangmiao and Shen, Chunhua and He, Tong},
  journal = {arXiv preprint arXiv:2507.13347},
  year    = {2025}
}
```
