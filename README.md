# Genetic Data Imputation

## Overview
This project aims to perform genetic data imputation for Allele1 and Allele2 using machine learning models. The code processes SNP and STR data, trains models, and generates predictions for missing genetic information.

## Features
- Data preprocessing for genetic data
- Training of machine learning models using LightGBM
- Model evaluation using Mean Absolute Error (MAE)
- Saving trained models for future predictions
- Docker support for easy deployment and environment management

## Prerequisites
- Python 3.9+
- Docker (optional, but recommended for containerized deployment)

## Project Structure

```
.
├── Dockerfile                              # Docker configuration
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── code/                                   # Directory containing all code
|    ├── imputation.py                      # Main script for performing genetic data imputation
|    ├── model_training.py                  # Script responsible for training the machine learning models for Allele1 and Allele2
|    └── utils.py                           # Utility functions for data preprocessing, encoding, and memory optimization
└── models/                                 # Directory to store trained machine learning models
|    ├── model_allele1.pkl                  # Trained model for predicting Allele1
|    ├── model_allele2.pkl                  # Trained model for predicting Allele2
|    ├── label_encoder_Name.pkl             # Label encoder for column Name
|    └── label_encoder_STR Name.pkl         # Label encoder for column STR Name
└── data/                                   # Directory containing all input data files required for the project
    ├── FinalReport.csv                     # Main genetic data report file containing SNP information           
    ├── snp_map_file.csv                    # Mapping file for SNP details and their characteristic           
    ├── STR_train.csv                       # Training dataset for STR (Short Tandem Repeat) information       
    ├── STR_test.csv                        # Test dataset for STR information used for model evaluation      
    └── STR_test_imputed.csv                # Output dataset with predicted Allele 1 and Allele 2
```

## Setup

### Open terminal

For Windows use PowerShell, for Linux and MacOS use Terminal

### Clone this repository

```
git clone https://github.com/SvetlanaDanilova/Genetic_Data.git
```

### Go to repository folder

```
cd Genetic_Data
```

### Add your data

Place the input .csv files to data folder as shown in project structure

### Build the Docker image

```
docker build -t <your-docker-image-name> .
```

### Run the Docker container
You can run the program by passing the option if you need to retrain models

```
docker run -it --rm -v "$(pwd):/app" --name <your-docker-container-name> <your-docker-image-name> <--retrain if need to retrain or --no-retrain if not need to retrain> 
```

### Output

After running the container, the resulting video with detected people and confidence scores will be saved in the output file you specified 

