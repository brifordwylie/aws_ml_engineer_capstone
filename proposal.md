# AWS ML Engineer Capstone Proposal

Welcome to the **PROPOSAL** for my capstone project for the Udacity AWS Machine Learning Course <https://www.udacity.com/enrollment/nd189>.

### Domain Background
For many modeling domains the prevalent perspective is that more data is always better. I'd like to investigate whether using just the **high quality** data is better than using **all** of the data for training and modeling.

Within the domain of drug discovery, training data often comes from bench experiments where a property like solubility is measured. As part of the data collection, there is often **systematic noise** where measurement variability is expected. There can also be anomalies associated with measurement errors or data entry issues. For all of these scenarios, we'd like to provide a set of data quality models that together will provide a data confidence score for each observation.

For this project, I'm going to use the AQSol public data as a representative dataset. For more information and details please see [Datasets and Inputs](#datasets-and-inputs).

### Problem Statement
There are modeling domains where the training data is inherently noisy. This noise may come from expected measurement variability or from errors in the measurement or issues with data entry. When this data is used for model training, we're often including **latent errors** in our training set without any sort of notification or visibility into the problematic observations. We propose that it's better to have less data of **high quality** than more data of unknown quality.

## Solution Statement

This project proposes the following solutions:

We'll use data confidence scoring to improve our modeling pipeline in two ways:

- Confidence scores for existing models: We'll provide a 'harness' where the predictions run on existing models can be given a complimentary confidence metric.
- High Quality Data Models: We'll construct a new model that only uses high quality data and compare the performance of that model to the existing model that uses all the data.

For this project, I'd like to create and deploy data quality models using AWS infrastructure. These models will provide various data quality metrics for feature space data distributions, neighborhood quality, and residuals.

1. **Quantile Regression + Residuals Model**: This model will provide predictions at different quantiles, offering a broader sense of the distribution within the training data. It will help in understanding the range and variability of predictions. Included as part of this functionality is the computation of residuals for all of the observations (using 5-Fold Cross validation and aggregating the results).
1. **K-Nearest Neighbors (KNN) Model**: This model will use the distances to the nearest neighbors in the feature space. Observations that have close neighbors with low variance in their target values will have higher neighborhood 'quality' levels, while those in high variance neighborhoods or sparse regions will have lower quality levels.

These models and endpoints will be implemented using AWS SageMaker, leveraging its robust infrastructure for training, deploying, and scaling machine learning models.

## Datasets and Inputs
For this project, we're going to use the publicly available AqSol Database. This is a curated reference set of aqueous solubility, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values of 9,982 unique compounds curated from 9 different publicly available aqueous solubility datasets. AqSolDB also contains some relevant topological and physico-chemical 2D descriptors. Additionally, AqSolDB contains validated molecular representations of each of the compounds.

Main Reference: <https://www.nature.com/articles/s41597-019-0151-1>

Data Download from the Harvard DataVerse: <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8>

## Benchmark Model

For the benchmark model, we'll use the AWS Linear Learner model. This model will provide point predictions without any data quality or confidence estimates, serving as a baseline to compare the performance against the evaluation metrics defined below.

## Evaluation Metrics

The performance of both the benchmark model and the proposed data quality models will be evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: To measure the accuracy of the point predictions. We'll measure the benchmark model with ALL the data (80/20 split), and then we'll partition the data into 3 segments: high, medium, and low quality. The MAE on 80/20 splits will be provided for the 4 categories: all, high, med, and low.
- **ROC Curves/AUC**: Similar to the MAE, we provide a ROC curve plot (with AUC scores) for each of the 4 categories (all, high, med, low). Since ROC curves are traditionally used with classification data, we'll employ an 'Accurate' column that looks at the residuals and if they are less than some delta then we'll consider the prediction as an accurate prediction. **Example:** For LogS solubility we might define any prediction within 0.5 log units as being an accurate prediction. This will be a parameter that can be adjusted for different use cases.

## Project Design

The project will develop and implement data quality/confidence models within the AWS SageMaker environment.

1. **AQSol Public Data**: We'll use the Aqueous Solubility public dataset to provide an example of realistic data that contains inherent noise in the target variable (Solubility).

1. **Model Training**: Will use the AWS SageMaker model training functionality to train our two data quality models.
   - **Quantile Regression + Residuals**: Model to predict different quantiles of the target variable. This model will also compute residuals as described above.
   - **K-Nearest Neighbors (KNN) Model**: Develop a KNN-based model to estimate confidence levels based on the density and variability of observations in the feature space.

1. **Endpoint Deployment**: Deploy the models on AWS SageMaker, setting up endpoints for real-time predictions and confidence estimates.
1. **Model Evaluation**: Evaluate the models using the defined metrics, comparing them against the benchmark model.
1. **Visualization and Reporting**: Create visualizations to illustrate the distribution of predictions and confidence levels, and compile a comprehensive report documenting the findings and insights from the project.

By following this workflow, the project aims to deliver robust data quality/confidence models that enhance the interpretability and reliability of machine learning predictions in AWS.