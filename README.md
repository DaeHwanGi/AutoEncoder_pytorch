## AutoEncoder with MNIST dataset

### Requirement
- python 3.6
- pytorch 0.4.1
- torchvision
- visdom
- numpy
- matplotlib

### Model preview
<center><img src="./image/GeneralAE.png"></center>

[GeneralAutoEncoer1](./General_AutoEncoder.ipynb)

<center><img src="./image/GeneralAE_vis.png"></center>

[GeneralAutoEncoer2](./General_AutoEncoder_vis.ipynb)

### MLP AutoEncoder result
#### simple model
![outcome0](./image/outcome0.png)
#### dimension reduction to 3d
![outcome1](./image/outcome1.png)
#### 3D scatter using matplotlib
![sd_scatter](./image/3d_scatter.png)
#### compare three axis (-0.5 ~ 0.5)
![comp_axis_0](./image/comp_axis_0.png)
![comp_axis_1](./image/comp_axis_1.png)
### Convolutional AutoEncoder result
#### simple model
![outcome2](./image/outcome2.png)
#### noise reduction
![outcome3](./image/outcome3.png)
#### Filter comparison
simple model | noise reduction
--- | ---
![filter1](./image/convFilter1.png) | ![filter2](./image/convFilter2.png)
