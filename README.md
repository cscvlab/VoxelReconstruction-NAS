# An Efficient End-to-End 3D Model Reconstruction based on Neural Architecture Search.
Yongdong Huang, Yuanzhan Li, Xulong Cao, Siyu Zhang, Shen Cai*, Ting Lu, and Yuqi Liu. An Efficient End-to-End 3D Model Reconstruction
based on Neural Architecture Search. Submitted to ICPR2022.

## Methodology
We complete the end-to-end neural network for 3D model reconstruction task by classifying binary voxels and utilizing the technology of neural architecture search (NAS).
Compared to other signed distance field (SDF) prediction or binary classification methods, our method achieves significantly higher reconstruction accuracy using fewer network parameters. 
![](IMGS/Fig1.png)
![](IMGS/Fig3.png)

[ONet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mescheder_Occupancy_Networks_Learning_3D_Reconstruction_in_Function_Space_CVPR_2019_paper.pdf) ,
[NI](https://arxiv.org/pdf/2009.09808v3.pdf) and
[NGLOD](https://openaccess.thecvf.com/content/CVPR2021/papers/Takikawa_Neural_Geometric_Level_of_Detail_Real-Time_Rendering_With_Implicit_3D_CVPR_2021_paper.pdf) are the methods we compared in our paper.

[ONet] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger, “Occupancy networks: Learning 3d reconstruction in function space,”, in CVPR, 2019.

[NI] Thomas Davies, Derek Nowrouzezahrai,  and Alec Jacobson,  “On the effectiveness ofweight-encoded neural implicit 3d shapes,” arXiv:2009.09808, 2020.

[NGLOD] Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles  Loop,  Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler, “Neural geometric level of detail:  real-time rendering with implicit 3d shapes,” in CVPR, 2021.



[//]: # ([3] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger,)

[//]: # (“Occupancy networks: Learning 3d reconstruction in function space,”)

[//]: # (in IEEE/CVF Conference on Computer Vision and Pattern Recognition)

[//]: # (&#40;CVPR&#41;, 2019, pp. 4455–4465.)

[//]: # ()
[//]: # ([4] T. Davies, D. Nowrouzezahrai, and A. Jacobson, “On the effectiveness of weight-encoded neural implicit 3d shapes,” arXiv preprint)

[//]: # (arXiv:2009.09808, 2020.)

[//]: # ()
[//]: # ([5] T. Takikawa, J. Litalien, K. Yin, K. Kreis, C. Loop, D. Nowrouzezahrai,)

[//]: # (A. Jacobson, M. McGuire, and S. Fidler, “Neural geometric level of)

[//]: # (detail: Real-time rendering with implicit 3d shapes,” in IEEE/CVF)

[//]: # (Conference on Computer Vision and Pattern Recognition &#40;CVPR&#41;, 2021,)

[//]: # (pp. 11 353–11 362.)

## Network
![](IMGS/Fig2.png)

## Experimental results
![](IMGS/Fig4.png)
![](IMGS/Table1.png)

## Result files
Models and reconstructions of the 3D objects presented in our paper are in the RESULTS folder. You can directly run eval.py to get the reconstruction results, or download [Meshlab](https://meshlab.en.softonic.com/) to open the reconstruction results provided by us. Note that when using MeshLab to view the reconstruction results we provide, select X Y Z for Point format and SPACE for Separator.

## Dataset
We use [Shapenet](https://shapenet.org/download/shapenetcore) and [Thingi10k](https://ten-thousand-models.appspot.com/) datasets, both of which are available from their official websites. [Thingi32]( https://github.com/nv-tlabs/nglod/issues/4) is composed of 32 simple shapes in Thingi10K. [ShapeNet150]( https://github.com/nv-tlabs/nglod/issues/4) contains 150 shapes in the ShapeNet dataset.

## Getting started

### Training
```bash
cd ./
bash train.sh
```

### Evaluation
```bash
python eval.py
```

### Ubuntu and CUDA version
We verified that it worked on ubuntu18.04 & cuda10.2.

### Python dependencies
```bash
python 3.6
tensorflow 2.0
```

## License
This project is licensed under the terms of the Apache License (see `LICENSE` for details).
