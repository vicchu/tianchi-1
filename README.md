# 天池-2019 年县域农业大脑AI挑战赛

## Dataset
The trainset contains two huge images, shape (50141, 47161, 3) & (46050, 77470, 3) each, and the labels contain 4 classes: 

crops    |   flue-cured tobacco   | corn |    myotonin    |  others 
---------|------------------------|------------------|----------------------|-----------  
labels     | 1 | 2 | 3 | 0  

we randomly crop it to 640*640 to train the model. 64 train samples are shown below.

<table>
  <tr>
    <td><img src="train_data.png?raw=true" width="1000"></td>
    <td><img src="train_label.png?raw=true" width="1000"></td>
  </tr>
</table>

## Devices
ubuntu14.01, CPU Memory : 32GB, GPU Memory : 11GB
