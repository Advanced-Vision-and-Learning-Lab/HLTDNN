
"""

# Downloading datasets:

Note: Due to the size of the datasets, the audio files were not 
upload to the repository. For the HLTDNN paper, only DeepShip dataset was used. 
Please follow the following instructions
to ensure the code works. If the dataset is used,
please cite the appropiate source (paper, repository, etc.) as mentioned
on the webpages and provided here.

##  DeepShip

Please download the [`DeepShip dataset`](https://github.com/irfankamboh/DeepShip) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `Deepship`
3. The structure of the `Deepship` folder is as follows:
```
[[`DeepShip dataset`](https://doi.org/10.1109/LGRS.2022.3156532)]
    ├── [['Cargo'](https://github.com/irfankamboh/DeepShip/tree/main/Cargo)]
    ├── [['Passenger'](https://github.com/irfankamboh/DeepShip/tree/main/Passengership)]
    ├── [['Tanker'](https://github.com/irfankamboh/DeepShip/tree/main/Tanker)]
    ├── [['Tug'](https://github.com/irfankamboh/DeepShip/tree/main/Tug)]
```
4. After download the dataset, please ['Get_preprocessed_data.py'](https://github.com/Peeples-Lab/HLTDNN/blob/master/Datasets/Get_preprocessed_data.py) file to generate the segments.
## <a name="CitingDeepShip"></a>Citing DeepShip

If you use the DeepShip dataset, please cite the following reference using the following entry.

**Plain Text:**

Irfan, M., Jiangbin, Z., Ali, S., Iqbal, M., Masood, Z., & Hamid, U. (2021). DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification. Expert Systems with Applications, 183, 115270.

**BibTex:**
```
@article{irfan2021deepship,
  title={DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification},
  author={Irfan, Muhammad and Jiangbin, Zheng and Ali, Shahid and Iqbal, Muhammad and Masood, Zafar and Hamid, Umar},
  journal={Expert Systems with Applications},
  volume={183},
  pages={115270},
  year={2021},
  publisher={Elsevier}
}

```
