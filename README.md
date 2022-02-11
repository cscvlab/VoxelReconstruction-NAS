# An Efficient End-to-End 3D Model Reconstruction based on Neural Architecture Search.
Yongdong Huang, Yuanzhan Li, Xulong Cao and Siyu Zhang, Shen Cai∗ , Ting Lu,Yuqi Liu.An Efficient End-to-End 3D Model Reconstruction
based on Neural Architecture Search.

##Methodology
we complete the end-to-end network by classifying binary voxels.
Compared to other signed distance field (SDF) prediction or
binary classification networks, our method achieves significantly
higher reconstruction accuracy using fewer network parameters.
![](IMGS/Fig1.png)
![](IMGS/Fig3.png)
##Network

![](IMGS/Fig2.png)
##Experiment
![](IMGS/Fig4.png)
![](IMGS/Table1.png)
##Results

##Dataset
We use [Shapenet](https://shapenet.org/download/shapenetcore) and [Thingi10k](https://ten-thousand-models.appspot.com/) datasets, both of which are available from their official website. [Thingi32]( https://github.com/nv-tlabs/nglod/issues/4) is composed of 32 simple shapes in Thingi10K. [ShapeNet150]( https://github.com/nv-tlabs/nglod/issues/4) contains 150 shapes in the ShapeNet dataset.
##ShapeNet



##Getting started

###Training
```bash
cd ./
bash train.sh
```

###Evaluation
```bash
python eval.py
```

###Ubuntu and CUDA version
We verified that it worked on ubuntu18.04 cuda10.2

###Python dependencies