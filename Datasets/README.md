# Downloading dataset:

Note: Due to the size of the datasets, the audio files were not 
uploaded to the repository. For the HLTDNN paper, DeepShip dataset was used. 
Please follow the following instructions
to ensure the code works. If the dataset is used,
please cite the appropiate source (paper, repository, etc.) as mentioned
on the webpage and provided here.

##  DeepShip

Please download the [`DeepShip dataset`](https://github.com/irfankamboh/DeepShip/issues/1)
and follow these instructions:

1. Create a folder called `Deepship`
2. Download the signals for each class in the `Deepship` folder:
        [`Cargo`](https://drive.google.com/drive/folders/1YyzrgY2tfFwtch3oTS29XUvKtEnsTgbw)
        [`Passengership`](https://drive.google.com/drive/folders/1aLn-XVaPYP8-RUzpS2SBDkGuNTWKtiNi)
        [`Tanker`](https://drive.google.com/drive/folders/1d-MrUfb8fPX8EmZIfVO5oBetVTxXfyOA)
        [`Tug`](https://drive.google.com/drive/folders/1b_gNLNammWm1HsRa3muLryccHQEAHDnT)
3. The structure of the `Deepship` folder is as follows:
```
Deepship/
    ├── Cargo/
    │   ├── Cargo1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Cargo2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    ├── Passenger/
    │   ├── Passenger1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Passenger2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    ├── Tanker/
    │   ├── Tanker1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Tanker2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    ├── Tug/
    │   ├── Tug1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Tug2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    └── ...
```

4. The dataset structure after segmentation is shown in the figure below, along with the number of segments in each class.      

[Note: The number of files for Cargo, Passengership, Tanker, and Tug is 109,191, 240, and 69, respectively after downloading from Google Drive.]
![Paper Figures](https://github.com/Peeples-Lab/HLTDNN/blob/master/Figures/Dataset.png)
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
