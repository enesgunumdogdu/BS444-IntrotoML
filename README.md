# BS444 - Introduction to Machine Learning Midterm Project

## Overview

This project investigates whether human blood microbiome data can serve as an indicator for cancer diagnosis. The project uses machine learning techniques to classify different types of cancer based on microbiome composition in blood samples.

## Dataset

The dataset contains blood sample data from 355 individuals with 4 different types of cancer:
- Colon cancer
- Breast cancer
- Lung cancer
- Prostate cancer

### Data Files

- **labels.csv**: Contains sample names and corresponding disease types for each individual
- **data.csv**: Contains microbiome data where each row represents:
  - Sample name (first column)
  - DNA fragment counts for each microorganism type (virus or bacteria) in subsequent columns
  - Total of 1,836 different microorganism features

## Methodology

### Data Preprocessing
- Data normalization using row-wise division by row sums
- Train-test split with 18.22% test size (random_state=14)

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: 
  - n_estimators: 100
  - random_state: 42

### Evaluation Metrics
- Accuracy
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)

## User Interface

The project includes a Tkinter-based graphical user interface that displays:
- Model accuracy
- Sensitivity metric
- Specificity metric

## Development Environment

- **Python Version**: 3.11.0
- **IDE**: Visual Studio Code
- **Key Libraries**:
  - pandas
  - scikit-learn
  - tkinter

## Usage

1. Ensure all required dependencies are installed:
```bash
pip install pandas scikit-learn
```

2. Place `data.csv` and `labels.csv` in the project directory

3. Run the project:
```bash
python mlproject.py
```

The application will:
- Load and preprocess the data
- Train the Random Forest classifier
- Display evaluation metrics in a GUI window

## Project Structure

```
BS444-IntrotoML/
├── mlproject.py      # Main project file
├── data.csv         # Microbiome data
├── labels.csv       # Sample labels and disease types
└── README.md        # Project documentation
```
