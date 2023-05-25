# Histogram_Layer_Target_Classification:
![Fig1_Workflow (1)](https://github.com/jarinritu/Test/assets/33591317/d0321e4c-e7fe-4a2f-9d2a-d9ba2ddba862)



**Histogram Layers for Improved Target Classification**

Jarin Ritu, Ethan Barnes, Riley Martell, Alexandra Van Dine and Joshua Peeples

Note: If this code is used, cite it: Authors TBD. Peeples-Lab/Histogram_Layer_Target_Classification: Initial Release (Version v1.0). 
Zenodo (TBD). https://doi.org/10.5281/zenodo.4404604 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4404604.svg)](https://doi.org/10.5281/zenodo.4404604)

[[`IEEE GRSL`](https://doi.org/10.1109/LGRS.2022.3156532)]

[[`arXiv`](https://arxiv.org/abs/2012.15764)]

[[`BibTeX`](#CitingHist)]

In this repository, we provide the paper and code for "Histogram Layers for Improved Target Classification."

## Installation Prerequisites

This code uses python, pytorch, and barbar. 
Please use [[`Pytorch's website`](https://pytorch.org/get-started/locally/)] to download necessary packages.
Barbar is used to show the progress of model. Please follow the instructions [[`here`](https://github.com/yusugomori/barbar)]
to download the module.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. To evaluate performance,
run `View_Results.py` (if results are saved out).

## Main Functions

The target classification code runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(**Parameters)```

2. Prepare dataset(s) for model
   
   ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```

## Parameters

The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/Peeples-Lab/Histogram_Layer_Target_Classification

└── root dir
    ├── demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameters file for demo.
    ├── Prepare_Data.py  // Load dataset for demo file.// Generate pytorch dataloader for each dataset.
    ├── View_Results.py // Run this after demo to view saved results[Generate results from the saved models]
  	└── Datasets  // Custom dataset and Dataset with original files
        ├── DeepShip // 4 classes[Cargo, Passenger, Tanker,Tug] with original audio files  
 	     ├── Get_preprocessed_data.py // Resampled the audio data generate segments for the dataset
        ├── DeepShipSegments.py  // Create Custom dataset for DeepShip.
        ├── Get_Audio_Features.py // Extract and Transform different features from the audio files
   └── Utils  //utility functions
        ├── Texture_information.py // Name and data directories for results script
        ├── Compute_FDR.py  // Compute Fisher Score
        ├── Confusion_mats..py  // Create and plot confusion matrix.
        ├── Generate_Learning_Curves..py  // Generate Learning Curves for training and validationt.
        ├── Generate_TSNE_visual.py  // Generate TSNE visuals 
        ├── Get_Optimizer.py  // Set of optimizer are defined to choose for.
        ├── Histogram_Model..py  // Load Histogram model with TDNN(or any backbone network)
        ├── Network_functions.py  // Contains functions to initialize, train, and test model.
        ├── RBFHistogramPooling.py  // Create Histogram layer.
        ├── Save_Results.py  // Save results from demo script.
        ├── pytorchtools.py  // Implement early stopping to terminate training based on validation metrics.
        ├── TDNN_MITLL.py  // Baseline TDNN model.

```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2023 TBD. All rights reserved.

## <a name="CitingHist"></a>Citing Histogram Layers for Improved Target Classification

If you use the target classification code, please cite the following reference using the following entry.

**Plain Text:**

TBD, "Histogram Layers for Improved Target Classification," in TBD.

**BibTex:**

```
TBD
```
