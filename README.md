# Histogram Layer Time Delay Neural Networks:
![Fig1_Workflow](https://github.com/Peeples-Lab/HLTDNN/blob/master/papers/Fig1_Workflow.png)



**HISTOGRAM LAYER TIME DELAY NEURAL NETWORKS FOR PASSIVE SONAR
CLASSIFICATION**

Jarin Ritu, Ethan Barnes, Riley Martell, Alexandra Van Dine and Joshua Peeples

Note: If this code is used, cite it: Jarin Ritu, Ethan Barnes, Riley Martell, Alexandra Van Dine and Joshua Peeples. Peeples-Lab/Histogram_Layer_Target_Classification: Initial Release (Version v1.0). 
Zenodo (TBD). https://doi.org/10.5281/zenodo.4404604 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4404604.svg)](https://doi.org/10.5281/zenodo.4404604)

[`IEEE WASPAA`][TBD]

[`arXiv`][TBD]
['BibTex'][TBD]




In this repository, we provide the paper and code for "Histogram layer time delay neural network for passive sonar classification."

## Installation Prerequisites


Please use [['requirements.txt'](https://github.com/Peeples-Lab/HLTDNN/blob/master/requirements.txt)] file and download necessary packages.

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
  	└── Datasets  // Custom dataset and dataset with original files
        ├── DeepShip // 4 classes[Cargo, Passenger, Tanker,Tug] with original audio files  
 	     ├── Get_preprocessed_data.py // Resampled the audio data generate segments for the dataset
        ├── DeepShipSegments.py  // Create Custom dataset for DeepShip.
        ├── Get_Audio_Features.py // Extract and transform different features from the audio files
   └── Utils  //utility functions
        ├── Texture_information.py // Name and data directories for results script
        ├── Compute_FDR.py  // Compute Fisher’s discriminant ratio
        ├── Confusion_mats..py  // Create and plot confusion matrix.
        ├── Generate_Learning_Curves..py  // Generate learning curves for training and validation.
        ├── Generate_TSNE_visual.py  // Generate TSNE visuals 
        ├── Get_Optimizer.py  // Set of optimizer are defined to choose for.
        ├── Histogram_Model..py  // Load histogram model with TDNN(or any backbone network)
        ├── Network_functions.py  // Contains functions to initialize, train, and test model.
        ├── RBFHistogramPooling.py  // Create histogram layer.
        ├── Save_Results.py  // Save results from demo script.
        ├── pytorchtools.py  // Implement early stopping to terminate training based on validation metrics.
        ├── TDNN_Model.py  // Baseline TDNN model.

```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2023 J. Ritu, E. Barnes, R. Martell, A. Dine and J. Peeples. All rights reserved.

## <a name="CitingHist"></a>Citing Histogram Layer Time Delay Neural Networks

If you use the target classification code, please cite the following reference using the following entry.

**Plain Text:**

J. Ritu, E. Barnes, R. Martell, A. Dine and J. Peeples, "HISTOGRAM LAYER TIME DELAY NEURAL NETWORKS FOR PASSIVE SONAR
CLASSIFICATION," submitted for review.

**BibTex:**

```
@inproceedings{Ritu2023histogram,
  title={Histogram layer time delay neural network for passive sonar classification},
  author={Ritu, Jarin and Barnes, Ethan, and Martell, Riley, and  Van Dine, Alexandra, and Peeples, Joshua},
  booktitle={TBD},
  pages={TBD},
  year={2023},
  organization={TBD}
}
```
