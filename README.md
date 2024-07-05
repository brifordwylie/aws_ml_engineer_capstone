# AWS ML Engineer Capstone Project

Welcome to my capstone project for the Udacity AWS Machine Learning Course <https://www.udacity.com/enrollment/nd189>.

### Project Description
Investigate confidence models using AWS. We use the functionality in AWS for model training, deployment, and running inference on the endpoints.

### Domain Background
For this project I'd like to create and deploy confidence models using AWS infrastructure. Many popular regression models provide point estimates but lack insights into the distribution and variability of the prediction. Within the domain of drug discovery training data comes from bench experiments where a property like solubility is measured. As part of the data collection there is often **systematic noise** where measurement variability is expected. There can also be anomalies associated from measurement errors or data entry issues. For all of these cases we'd like a confidence model that's data-centric and provides a generalized data confidence score for each observation.

For this project I'm going to use the AQSol public data as a representative dataset. For more information and details please see [Datasets and Inputs](#datasets-and-inputs)

### Problem Statement

(needs work) This project aims to develop and implement two confidence models: quantile regression to capture the distribution of the training data and KNN to provide confidence levels for individual observations.

## Solution Statement

This project proposes the following solutions:

1. **Quantile Regression Model**: This model will provide predictions at different quantiles, offering a broader sense of the distribution within the training data. It will help in understanding the range and variability of predictions.
2. **K-Nearest Neighbors (KNN) Confidence Model**: This model will use the distances to the nearest neighbors in the feature space to assign confidence levels to individual predictions. Observations that have close neighbors with low variance in their target values will have higher confidence levels, while those in high variance neighborhoods or sparse regions will have lower confidence levels.

These models will be implemented using AWS SageMaker, leveraging its robust infrastructure for training, deploying, and scaling machine learning models.

## Datasets and Inputs
For this project we're going to use the publicly available AqSol Database. This is a curated reference set of aqueous solubility, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values of 9,982 unique compounds curated from 9 different publicly available aqueous solubility datasets. AqSolDB also contains some relevant topological and physico-chemical 2D descriptors. Additionally, AqSolDB contains validated molecular representations of each of the compounds.

Main Reference: <https://www.nature.com/articles/s41597-019-0151-1>

Data Dowload from the Harvard DataVerse: <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8>


## Benchmark Model

A traditional regression model (using an XGBoost Regressor) will be used as the benchmark. This model will provide point predictions without confidence estimates, serving as a baseline to compare the performance and added value of the proposed confidence models.

## Evaluation Metrics

The performance of both the benchmark model and the proposed confidence models will be evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: To measure the accuracy of the point predictions.
- **Coverage Probability**: To assess how well the predicted quantiles cover the actual outcomes.
- **Calibration**: To evaluate the alignment between predicted confidence levels and observed frequencies.

These metrics are appropriate given the context of the problem and will provide a comprehensive evaluation of the models' performance.

## Presentation

The proposal follows a structured format, ensuring clarity and conciseness. Each section is specifically tailored to provide relevant information about the project. Proper citations will be included for all referenced academic research and resources.

## Project Design

The project will follow a systematic workflow to develop and implement the confidence models:

1. **Data Collection and Preprocessing**: Acquire and preprocess the dataset, handling missing values, encoding categorical variables, and normalizing features.
2. **Model Development**:
   - Train a Quantile Regression model to predict different quantiles of the target variable.
   - Develop a KNN-based model to estimate confidence levels based on the density of observations in the feature space.
3. **Model Evaluation**: Evaluate the models using the defined metrics, comparing them against the benchmark model.
4. **Deployment**: Deploy the models on AWS SageMaker, setting up endpoints for real-time predictions and confidence estimates.
5. **Visualization and Reporting**: Create visualizations to illustrate the distribution of predictions and confidence levels, and compile a comprehensive report documenting the findings and insights from the project.

By following this workflow, the project aims to deliver robust confidence models that enhance the interpretability and reliability of machine learning predictions in AWS.

