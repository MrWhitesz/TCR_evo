# TCR_evo

## Overview
TCR_evo is an innovative platform that integrates experimental validation data from mammalian cell surface display and deep sequencing with deep learning frameworks for high-throughput T-cell receptor (TCR) engineering. The platform was initially trained and optimized using unique experimental data from single-point and two-point mutation libraries, ultimately enabling accurate prediction of complex triple- and quadruple-point TCR mutations. This project specifically focuses on predicting TCR variants with enhanced binding to the influenza epitope 'HLA-A2-GILGFVFTL', demonstrating the platform's ability to identify optimized TCR sequences through iterative learning from experimental feedback.

## Installation

### Requirements
- fair-esm==2.0.0
- numpy
- scikit-learn==1.3.0
- torch==1.11.0
- tqdm==4.66.1

```bash
pip install -r requirements.txt
```

## Usage
### Datasets
- One-point-mutaion experiement data: 
1. original: `train_data/experiment_data.csv`
2. processed: `train_data/exp_train_fullseq_th0.csv`
- Two-point-mutaion experiement data: 
1. original: `test_data/exp_2point_mut.csv`
2. processed: `test_data/twoPoint_mut_th0.csv`
- Retraining data for other models:  `train_data/used_train_data.csv`
- Testing data: `test_data/G9L_test_len8-18.csv`

### Training the Models
```python
# Example for training
python model_train_transformer_onetwoMutExp_th0.py
```


## Citation
If you use this code or find our work helpful, please cite:
[Citation information to be added]
