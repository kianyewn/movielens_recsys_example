# Movie Recommendation ML Pipeline

A production-grade machine learning pipeline for processing user-movie ratings data and generating personalized movie recommendations.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Components](#pipeline-components)
- [Usage Guide](#usage-guide)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Development](#development)

## Overview

This pipeline implements an end-to-end recommendation system with:
- Data processing for users, movies, and ratings
- Feature engineering with numerical and categorical encoders
- LightGBM ranking model training with hyperparameter optimization
- MLflow experiment tracking
- Model evaluation with ranking metrics (NDCG, MAP, etc.)
- Inference pipeline for generating recommendations

## Project Structure

```
.
├── configs/                  # Configuration files
│   ├── data_config.py       # Data pipeline settings
│   ├── lgbm_config.py       # LightGBM model parameters  
│   └── model_config.py      # General model settings
├── src/
│   ├── data_processing/     # Data preprocessing
│   │   ├── movies.py        # Movie data processing
│   │   ├── ratings.py       # User ratings processing
│   │   └── users.py         # User data processing
│   ├── feature_processing/  # Feature engineering
│   │   ├── encoders.py      # Feature encoders
│   │   └── process_features.py  # Feature pipeline
│   ├── models/             # Model implementations
│   │   └── lgbm/           # LightGBM ranking model
│   └── pipelines/          # Main execution pipelines
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Components

### 1. Data Processing
The pipeline processes three main data types:
- User data: Demographic information
- Movie data: Movie metadata and features
- Ratings data: Historical user-movie interactions

### 2. Feature Engineering
- Numerical feature scaling
- Categorical encoding
- Negative sampling for training
- Ranking preparation

### 3. Model
- LightGBM ranking model
- Hyperparameter optimization with Optuna
- MLflow experiment tracking
- Evaluation metrics: NDCG, MAP, Precision, Recall

## Usage Guide

### Data Processing & Training

1. Process raw training data:
```bash
python src/pipelines/main_data_processing_training.py
```

2. Generate features for training:
```bash
python src/pipelines/main_feature_processing_training.py
```

3. Train the model:
```bash
python src/pipelines/main_model_training.py
```
This will:
- Perform hyperparameter optimization using Optuna
- Train both default and optimized models
- Track experiments with MLflow
- Save the models to configured paths

4. Evaluate model performance:
```bash
python src/pipelines/main_model_evaluation.py
```

### Inference Pipeline

1. Process new data:
```bash
python src/pipelines/main_data_processing_inference.py
```

2. Generate features for inference:
```bash
python src/pipelines/main_feature_processing_inference.py
```

3. Generate recommendations:
```bash
python src/pipelines/main_model_inference.py
```

### Model Retraining

To retrain the model with new data:

1. Process new data using the inference data processing pipeline
2. Update the training dataset configuration in `configs/data_config.py`
3. Run the training pipeline with the updated data

## Configuration

### Data Configuration
Edit `configs/data_config.py` to configure:
- Input/output paths
- Data processing parameters
- Feature selection

### Model Configuration
Edit `configs/lgbm_config.py` to modify:
- Model hyperparameters
- Training settings
- Evaluation metrics
- MLflow experiment settings

## Development

### Adding New Features

1. Add feature logic in `src/feature_processing/encoders.py`:
```python
class NewFeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Feature fitting logic
        return self

    def transform(self, X, y=None):
        # Feature transformation logic
        return X_transformed
```

2. Update `ProcessFeatures` class in `process_features.py`
3. Update model configuration to include new features

### Model Evaluation

The pipeline evaluates models using ranking metrics:
- NDCG@k
- MAP@k
- Precision@k
- Recall@k

View results in:
- MLflow experiment tracking UI
- Saved evaluation results
- Learning curves generated during training

## Troubleshooting

### Common Issues

1. Memory Issues
```python
# Reduce batch size in config
MODEL_CONFIG = {
    'batch_size': 1024  # Reduce if OOM
}
```

2. Feature Processing Errors
- Check data types match between training and inference
- Verify all required columns are present
- Ensure encoders are properly saved/loaded

3. Model Performance
- Review feature importance scores
- Check training/validation curves for overfitting
- Validate data quality and preprocessing steps

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

[Add your license information here]
```