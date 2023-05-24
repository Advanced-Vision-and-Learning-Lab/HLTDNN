# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. For the DREN paper, only KTH-TIPS-2b, DTD, and PRMI were used. We also have
two large scale datasets, GTOS-mobile and MINC-2500, implemented for the code.
Please follow the following instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

##  KTH Textures under varying Illumination, Pose and Scale (KTH-TIPS-2b) [[`BibTeX`](#CitingKTH)]

Please download the [`KTH-TIPS-2b dataset`](https://www.csc.kth.se/cvap/databases/kth-tips/download.html) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `KTH-TIPS2-b`
3. The structure of the `KTH-TIPS2-b` folder is as follows:
```
└── root dir
    ├── Images   // Contains folders of images for each class.
    ├── kth-tips2.pdf   // documentation of the dataset
    ├── kth-tips-2b-sizes.txt   // lists all images in the dataset and their sizes
    ├── README.txt   // README file from KTH-TIPS curators

```
## <a name="CitingKTH"></a>Citing KTH-TIPS-2b

If you use the KTH-TIPS-2b dataset, please cite the following reference using the following entry.

**Plain Text:**

Tavakoli Targhi, A., Hayman, E., Caputo, B., Mallikar-juna, P., Fritz, M., and  Eklundh, J.O.
“The kth-tips2 database,” KTH Royal Institute of Technology, 2006.

**BibTex:**
```
@article{mallikarjuna2006kth,
  title={The kth-tips2 database},
  author={Mallikarjuna, P and Targhi, Alireza Tavakoli and Fritz, Mario and Hayman, Eric and Caputo, Barbara and Eklundh, Jan-Olof},
  journal={KTH Royal Institute of Technology},
  year={2006}
}

```
##  Describable Texture Dataset (DTD) [[`BibTeX`](#CitingDTD)]

Please download the [`DTD dataset`](https://www.robots.ox.ac.uk/~vgg/data/dtd/) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `DTD`
3. The structure of the `DTD` folder is as follows:
```
└── root dir
    ├── images   // Contains folders of images for each class.
    ├── imdb // Not used.
    ├── labels  // Contains training,validation, and test splits.   
```
## <a name="CitingDTD"></a>Citing DTD

If you use the DTD dataset, please cite the following reference using the following entry.

**Plain Text:**

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014). 
Describing textures in the wild. In Proceedings of the IEEE Conference on 
Computer Vision and Pattern Recognition (pp. 3606-3613).

**BibTex:**
```
@InProceedings{cimpoi14describing,
	     Author    = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
	     Title     = {Describing Textures in the Wild},
	     Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
	     Year      = {2014}}
```
##  Plant Root Minirhizotron Imagery (PRMI) [[`BibTeX`](#CitingPRMI)]

Please download the [`PRMI dataset`](https://gatorsense.github.io/PRMI/) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `PRMI`
3. The structure of the `PRMI` folder is as follows:
```
└── root dir
    ├── val   // Validation data.
        ├── images   // Input images.
        ├── labels_image_gt   // Metadata for each image.
        ├── masks_pixel_gt   // Pixel label ground truth masks.
    ├── train // Training data.
        ├── images   // Input images.
        ├── labels_image_gt   // Metadata for each image.
        ├── masks_pixel_gt   // Pixel label ground truth masks.
    ├── test  // Test data. 
        ├── images   // Input images.
        ├── labels_image_gt   // Metadata for each image.
        ├── masks_pixel_gt   // Pixel label ground truth masks.  
```
## <a name="CitingPRMI"></a>Citing PRMI

If you use the PRMI dataset, please cite the following reference using the following entry.

**Plain Text:**

W. Xu, G. Yu, Y. Cui, R. Gloaguen, A. Zare, J. Bonnette, J. Reyes-Cabrera, A. Rajurkar, D. Rowland, R. Matamala, 
J. Jastrow, T. Juenger, and F. Fritschi. “PRMI: A Dataset of Minirhizotron Images for Diverse Plant Root Study.” 
In AI for Agriculture and Food Systems (AIAFS) Workshops at the AAAI conference on artificial intelligence. 
February, 2022.

**BibTex:**
```
@misc{xu2022prmi,
      title={PRMI: A Dataset of Minirhizotron Images for Diverse Plant Root Study}, 
      author={Weihuang Xu and Guohao Yu and Yiming Cui and Romain Gloaguen and Alina Zare and Jason Bonnette 
      and Joel Reyes-Cabrera and Ashish Rajurkar and Diane Rowland and Roser Matamala and Julie D. Jastrow 
      and Thomas E. Juenger and Felix B. Fritschi},
      year={2022},
      eprint={2201.08002},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Extension of the Ground Terrain in Outdoor Scenes (GTOS-mobile) [[`BibTeX`](#CitingGTOS_m)]

Please download the 
[`GTOS-mobile dataset`](https://github.com/jiaxue1993/Deep-Encoding-Pooling-Network-DEP-) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `gtos-mobile`
3. The structure of the `gtos-mobile` folder is as follows:
```
└── root dir
    ├── test   // Contains folders of test images for each class.
    ├── train // Contains folders of training images for each class.  
```
## <a name="CitingGTOS_m"></a>Citing GTOS-mobile

If you use the GTOS-mobile dataset, please cite the following reference using the following entry.

**Plain Text:**

Xue, J., Zhang, H., & Dana, K. (2018). Deep texture manifold for ground 
terrain recognition. In Proceedings of the IEEE Conference on Computer Vision 
and Pattern Recognition (pp. 558-567).

**BibTex:**
```
@inproceedings{xue2018deep,
  title={Deep texture manifold for ground terrain recognition},
  author={Xue, Jia and Zhang, Hang and Dana, Kristin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2018}
}
```
## Subset of Material in Context (MINC-2500) [[`BibTeX`](#CitingMINC)]

Please download the 
[`MINC-2500 dataset`](http://opensurfaces.cs.cornell.edu/publications/minc/) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `minc-2500`
3. The structure of the `minc-2500` folder is as follows:
```
└── root dir
    ├── images   // Contains folders of images for each class.
    ├── labels // Contains training,validation, and test splits.
    ├── categories.txt  // Class names for dataset
    ├── README.txt  // README file from MINC-2500 curators
       
```
## <a name="CitingMINC"></a>Citing MINC-2500

If you use the MINC-2500 dataset, please cite the following reference using the following entry.

**Plain Text:**

Bell, S., Upchurch, P., Snavely, N., & Bala, K. (2015). Material recognition 
in the wild with the materials in context database. In Proceedings of the IEEE 
conference on computer vision and pattern recognition (pp. 3479-3487).

**BibTex:**
```
@inproceedings{bell2015material,
  title={Material recognition in the wild with the materials in context database},
  author={Bell, Sean and Upchurch, Paul and Snavely, Noah and Bala, Kavita},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3479--3487},
  year={2015}
}
```
