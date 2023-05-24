# Histogram_Layer_Target_Classification:

**Histogram Layers for Improved Target Classification**

Ethan Barnes

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
    ├── Prepare_Data.py  // Load data for demo file.
    ├── Texture_Information.py // Class names and directories for datasets.
    ├── View_Results.py // Run this after demo to view saved results.
    ├── papers  // Related publications.
    │   ├── readme.md //Information about paper
    └── Utils  //utility functions
        ├── Compute_FDR.py  // Compute Fisher Discriminant Ratio for features.
        ├── Confusion_mats.py  // Generate confusion matrices.
        ├── Generate_Histogram_Vid.py  // Generates a video showing how the histogram layer varies with each epoch
        ├── Generate_Learning_Curves.py  // Generates the learning curves for the model
        ├── Generate_TSNE_visual.py  // Generate TSNE visualization for features.
        ├── Histogram_Model.py  // Generate HistRes_B models.
        ├── Network_functions.py  // Contains functions to initialize, train, and test model. 
        ├── Plot_Accuracy.py // Plots the average and std of metrics for each model  
        ├── RBFHistogramPooling.py  // Create histogram layer. 
        ├── Save_Results.py  // Save results from demo script.
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