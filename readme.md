# SQLshield

A deep learning-based Web Application Firewall (WAF) that detects SQL injection attacks using Character-level Convolutional Neural Networks (CNN).

## Overview

SQLshield is a machine learning project that provides real-time SQL injection detection capabilities. It uses a character-level CNN to analyze SQL queries and classify them as either safe or malicious. The project includes both training infrastructure and a working Flask demo that showcases the WAF in action.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.9.1
- Flask 3.1.2
- pandas, scikit-learn, numpy

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

Train a new model from scratch using your dataset:

```bash
python3 train.py --path Modified_SQL_Dataset.csv
```


### Running the Demo

Start the Flask application to see SQLshield in action:

```bash
python3 -m demo.app
```

Navigate to `http://localhost:5000` to access the demo interface.

**Demo Features:**
- **Vulnerable Endpoint** (`/vulnerable`): Standard SQL query execution without protection
- **Secure Endpoint** (`/secure`): Protected by SQLshield WAF decorator
- **Test Login**: Username: `testuser`, Password: `5dccc7bf2d9d9897c411e7d4b1b99480`

**Try SQL Injection:**
```
# Vulnerable endpoint will execute this attack
id: ' OR '1'='1
pw: ' OR '1'='1

# Secure endpoint will block this attack
Response: {"error": "Invalid input detected"}
```

## Dataset

**Source**: [Kaggle SQL Injection Dataset](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset/data)

## Model Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 64 |
| Conv Filters | 32 |
| Kernel Size | 4 |
| Dropout Rate | 0.6 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Batch Size | 64 |

### Performance

The model is trained with cross-entropy loss and reports:
- Training accuracy per epoch
- Validation accuracy per epoch
- Training/validation loss per epoch

Pre-trained weights are available at: `https://cdn.xdcs.me/sqlshield.pth`

## Dataset Attribution

Dataset provided by Sajid576 on Kaggle:  
https://www.kaggle.com/datasets/sajid576/sql-injection-dataset/data
