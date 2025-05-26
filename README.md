# AWS ML Engineer Capstone Project

Welcome to my capstone project for the Udacity AWS Machine Learning Course <https://www.udacity.com/enrollment/nd189>.

## Project Overview
For this project we're going to explore ways of computing confidence for regression model predictions. In particular we'd like to deploy 'sets of models' into AWS, ensembles for bootstrapping and quantile regression. Our model scripts and AWS endpoints will use groups of models and provide additional information like prediction intervals and quantile ranges. We're also going to compare/contrast these against a KNN model for regression confidence metrics.

## Datasets and Inputs
For this project, we're going to use the publicly available AqSol Database [1]. This is a curated reference set of aqueous solubility, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values of 9,982 unique compounds curated from 9 different publicly available aqueous solubility datasets. AqSolDB also contains some relevant topological and physico-chemical 2D descriptors. Additionally, AqSolDB contains validated molecular representations of each of the compounds.

Data Download from the Harvard DataVerse: <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8>


## Problem Statement
Within the domain of **drug discovery**, data used for model training often comes from bench experiments where a property like solubility is measured. As part of the data collection, there is often **noise** associated with the measurement, differences in temperature, pH, and extraction procedures can lead to variability. Also compounds can hit activity cliffs (see [2][3][4]) where a small molecular change can lead to a significant difference in the response variable. For all of these scenarios, we'd like to provide a set of models that together will provide both a regression prediction and confidence score for each observation.

## Metrics
We'd like our confidence metric to capture areas in feature space where the model is more or less confident in it's predictions. In areas of high confidence we'd expect to have smaller errors and in areas of low confidence we'd expect to see larger errors. We'll use statistics and boxplots that capture the prediction errors.

The metrics will include standard descriptive statistics about each of the confidence 'bands' (low, medium, and high). 

- min/max
- mean
- stddev
- Q1, Q2, Q3, IQR

The delination of the confidence bands (low, medium, and high) will be as follows:

- **Low** (0.0 to 0.33)
- **Medium** (0.33 to 0.66)
- **High** (0.66 to 1.0)

The calculation of the confidence score will be domain specific, meaning that for our particular dataset and target values will look at the ensemble model outputs and determine a heuristic for a confidence score that ranges from 0.0 to 1.0.

# Analysis

## Data Exploration
**Dataset:** For this project, we're using the AQSol public data as a representative dataset. For more information and details please see [Datasets and Inputs](#datasets-and-inputs).

### Data Statistics
- **Rows:** 9982
- **Target Column:** solubility
- **Feature Columns:** 17
- **Features:** molwt, mollogp, molmr, heavyatomcount, numhacceptors, numhdonors, numheteroatoms, numrotatablebonds, numvalenceelectrons, numaromaticrings, numsaturatedrings, numaliphaticrings, ringcount, tpsa, labuteasa, balabanj, bertzct


**Target: solubility**
<table>
<tr>

<td>
<img src="images/solubility_boxplot.png" alt="sol_box_plot" width="1200"/>
</td>
<td>

```
fs.descriptive_stats()["solubility"]

Out[36]:
{'min': -13.1719,
 'q1': -4.318666603641981,
 'median': -2.6081432110925657,
 'q3': -1.213113009169698,
 'max': 2.1376816201,
 'mean': -2.8899088047869865,
 'stddev': 2.368154448407012}
``` 
 
</td>

</tr>
</table>

**Feature Distributions using Violin Plots**
<img src="images/feature_distributions.png" alt="sol_box_plot" width="1200"/>

**Sample Rows**

```
[●●●]Workbench:scp_sandbox> df[comp_columns]
Out[41]:
      solubility    molwt  mollogp     molmr  heavyatomcount  numhacceptors  ...  ringcount    tpsa   labuteasa      balabanj      bertzct  solubility_class
0      -5.921700  197.064  4.14660   53.9680            12.0            0.0  ...        2.0    0.00   80.719600  2.797846e+00   384.358135               low
1      -5.984242  236.270  3.02440   69.1270            18.0            2.0  ...        3.0   34.14  105.265479  2.268429e+00   668.547220               low
2      -6.384617  242.447  5.72400   77.5710            17.0            1.0  ...        0.0    9.23  109.326928  2.846262e+00   108.693586               low
3      -3.591300  421.422  1.62540   90.6954            27.0            5.0  ...        3.0  118.36  153.035266  2.108091e+00  1085.280492              high
4      -2.800100  100.014  1.99100   11.4580             6.0            0.0  ...        0.0    0.00   31.076732  3.675949e+00    55.617271              high
...          ...      ...      ...       ...             ...            ...  ...        ...     ...         ...           ...          ...               ...
9977   -3.951281  269.059  1.66150   56.1384            16.0            5.0  ...        1.0   74.44  100.246449  2.931415e+00   428.234717              high
9978   -1.657600  146.149  1.33540   41.2028            11.0            3.0  ...        2.0   46.01   63.346597  2.913161e+00   381.253771              high
9979   -1.020815  482.584  3.06140  121.8500            32.0           10.0  ...        3.0  127.73  190.866146  5.344871e-07  1146.856409              high
9980    0.938623  115.180  0.04437   35.1007             8.0            1.0  ...        0.0   30.33   50.404038  3.651234e+00    75.674572              high
9981   -6.964446  368.558  6.58950  107.4476            26.0            2.0  ...        0.0   74.60  159.624944  3.369474e+00   376.338537               low
```


## Exploratory Visualization
Our main goal here is to use AWS **ensemble** models to help us identify areas of feature space where the model has high or low confidence in its predictions. To help us visualize the feature space we'll use the UMAP[5] projection algorithm to project the 17 dimenional feature space down to 2.


<figure>
  <img src="images/umap_sol.png" alt="UMAP solubility plot" width="1200"/>
  <figcaption><em>UMAP 2D projection of 17 dimensional feature space showing logS solubility of each compound (n=9982)</em></figcaption>
</figure>

In the image above we can see that some areas have a relatively low target variance and standard regression models (like XGBRegressor) should be able to make relatively accurate predictions in those areas. We also some some areas with higher variance that may indicate compounds on activity cliffs, noisy experimental conditions, or simply erroneous solubility measurements.

**Using Solubility Classification Colors**

Although this project will strictly be using regression models, here we want to better illuminate the areas of feature space that have high variance by coloring the plot above with a "high, medium, and low" solubility values. For logS solubility those values are traditionally based on these ranges:

- **High Solubility**: > -4 logS
- **Medium Solubility:** > -5 and < -4 logS
- **Low Solubility:** < -5 logS

<figure>
  <img src="images/umap_sol_class.png" alt="UMAP solubility plot" width="1200"/>
  <figcaption><em>Same UMAP projection as above but colored by solubility classes to better illustrate areas of high variance</em></figcaption>
</figure>


<figure>
  <img src="images/showing_zoom_in.png" alt="UMAP solubility plot" width="1200"/>
  <figcaption><em>Zooming into an area with high variability. Compounds show solubility from every category("low", "medium", and "high")</em></figcaption>
</figure>


## Algorithms and Techniques

1. **Prediction Intervals using Bootstrapping**
   - Bootstrapping involves repeatedly sampling from the training data and fitting the model multiple times to generate a distribution of predictions. This can be used to estimate prediction intervals. This approach should be robust and doesn't make assumptions about a particular distribution. 
  
1. **Quantile Regression**
   - A set of models with different objective functions that can provide quantile estimates for predictions, offering a broader sense of the distribution within the training data. It will help in understanding the range and variability of predictions. We'll estimate a range of quantiles that should give us a 'spread' of target values within that region of feature space.

1. **K-Nearest Neighbors (KNN) Model**: This model will use the distances to the nearest neighbors in feature space. Observations that have close neighbors with low variance in their target values will have higher confidence values, while those in high variance neighborhoods or sparse regions will have lower confidence.

These models and endpoints will be implemented using AWS SageMaker, leveraging its robust infrastructure for training, deploying, and scaling machine learning models.



## Benchmark Model

In this case we're adding a confidence score to our regression model. For the benchmark model, we'll use a standard XGBRegressor() model. This model will provide point predictions without any confidence estimates, serving as a baseline. Our metrics of comparison will include:

- **Mean Absolute Error (MAE)**: To measure the accuracy of the point predictions. We'll measure the benchmark model with ALL the data (80/20 split), and then we'll partition the data into 3 segments: high, medium, and low quality. The MAE on 80/20 splits will be provided for the 4 categories: all, high, med, and low.

- **Residual Box Plots**: To provide an overview of the predictions within our confidence levels we'll show box plots of the residuals for each confidence level. The box plots should show tighter bounds on the residuals for areas of high confidence.
  
- **(Maybe) ROC Curves/AUC**: Similar to the MAE, we provide a ROC curve plot (with AUC scores) for each of the 4 categories (all, high, med, low). Since ROC curves are traditionally used with classification data, we'll employ an 'Accurate' column that looks at the residuals and if they are less than some delta then we'll consider the prediction as an accurate prediction. **Example:** For LogS solubility we might define any prediction within 0.5 log units as being an accurate prediction. This will be a parameter that can be adjusted for different use cases.

# Methodology

## Data Preprocessing

We're using the Aqueous Solubility public dataset to provide an example of realistic data that contains inherent noise in the target variable (Solubility). We first pushed the CSV file onto an S3 bucket and then loaded the data into AWS Athena, making sure that the types were correct and running a few queries.


<figure>
  <img src="images/aqsol_data_in_athena.png" alt="sol_box_plot" width="1200"/>
  <figcaption><em>AQSol Data in AWS Athena</em></figcaption>
</figure>

Once the data is available through AWS Athena we can query it directly from Python using AWSWrangler.

```
df = wr.athena.read_sql_query(
    sql="select * from aqsol_data",
    database=database,
    ctas_approach=False,
    boto3_session=my_boto3_session,
)

        id                                               name  ...      bertzct solubility_class
0      A-3         N,N,N-trimethyloctadecan-1-aminium bromide  ...   210.377334             high
1      A-4                           Benzo[cd]indol-2(1H)-one  ...   511.229248             high
2      A-5                               4-chlorobenzaldehyde  ...   202.661065             high
3      A-8  zinc bis[2-hydroxy-3,5-bis(1-phenylethyl)benzo...  ...  1964.648666             high
4      A-9  4-({4-[bis(oxiran-2-ylmethyl)amino]phenyl}meth...  ...   769.899934           medium
...    ...                                                ...  ...          ...              ...
9977  I-84                                         tetracaine  ...   374.236893             high
9978  I-85                                       tetracycline  ...  1148.584975             high
9979  I-86                                             thymol  ...   251.049732             high
9980  I-93                                          verapamil  ...   938.203977             high
9981  I-94                                           warfarin  ...   909.550973           medium
```
We can also inspect all the data types to make sure our S3 data and load into Athena worked correctly.

```
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9982 entries, 0 to 9981
Data columns (total 27 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   id                   9982 non-null   string
 1   name                 9982 non-null   string
 2   inchi                9982 non-null   string
 3   inchikey             9982 non-null   string
 4   smiles               9982 non-null   string
 5   solubility           9982 non-null   float64
 6   sd                   9982 non-null   float64
 7   ocurrences           9982 non-null   Int64
 8   group                9982 non-null   string
 9   molwt                9982 non-null   float64
 10  mollogp              9982 non-null   float64
 11  molmr                9982 non-null   float64
 12  heavyatomcount       9982 non-null   float64
 13  numhacceptors        9982 non-null   float64
 14  numhdonors           9982 non-null   float64
 15  numheteroatoms       9982 non-null   float64
 16  numrotatablebonds    9982 non-null   float64
 17  numvalenceelectrons  9982 non-null   float64
 18  numaromaticrings     9982 non-null   float64
 19  numsaturatedrings    9982 non-null   float64
 20  numaliphaticrings    9982 non-null   float64
 21  ringcount            9982 non-null   float64
 22  tpsa                 9982 non-null   float64
 23  labuteasa            9982 non-null   float64
 24  balabanj             9982 non-null   float64
 25  bertzct              9982 non-null   float64
 26  solubility_class     9982 non-null   string
dtypes: Int64(1), float64(19), string(7)
memory usage: 2.1 MB
```

### Descriptive Statistics
The AWS Athena SQL engine allows you to make queries for descriptive statistics (like box plots). Here an example:

```
query = 'SELECT min("solubility") AS "min", 
         approx_percentile("solubility", 0.25) AS "q1", 
         approx_percentile("solubility", 0.5) AS "median",
         approx_percentile("solubility", 0.75) AS "q3",
         max("solubility") AS "max" from aqsol_data'

ds.query(query)
Out[13]:
       min        q1    median        q3       max
0 -13.1719 -4.332174 -2.611036 -1.194223  2.137682
```
We use these Athena queries to create these distribution plots using Dash/Plotly [6][7]. They are called violin plots and use a combination of the 'box-plot' queries above and smart-sample to create each plot.

<figure>
  <img src="images/violin_explanation.png" alt="sol_box_plot" width="1200"/>
  <figcaption><em>Example of how we use Athena Queries to create the data for Violin Plots in Plotly</em></figcaption>
</figure>

<figure>
  <img src="images/feature_distributions.png" alt="sol_box_plot" width="1200"/>
  <figcaption><em>Violin Plots of the distributions of the target variable (solubility) and features of the AQSol Public Dataset </em></figcaption>
</figure>


## Implementation

The project has developed a set of regression models within the AWS SageMaker environment. In addition to point predictions these models provide additional data about distributions and prediction intervals that allow us to assign a confidence metric to each regression prediction.

### AWS Model Script Overview
AWS Model Scripts have a general set of entry points that AWS uses for both **training** the model and running **inference** on the deployed AWS endpoint. Note: After training and deploying lots of AWS models, I've standardized on using Pandas DataFrames for 'data interchange' points.

```
# The main function is used during the **training** of models
if __name__ == "__main__":

    
    # Typically use argparse/env vars to pull in 
    # - SM_MODEL_DIR (where to put your model)
    # - SM_CHANNEL_TRAIN (where to get your training data)
    # - SM_OUTPUT_DATA_DIR (optional: output data like validation predictions, etc)
    
    # Main Training
    # - Read in Training Data
    # - Train model: model.fit(X_train, y_train)
    # - Validation metrics
    # - Save Model (also label encoders, feature arrays, etc)


# The rest of the functions are used for Endpoint Inference
def model_fn(model_dir):
    """Deserialized and return model(s) from the model directory."""

def input_fn(input_data, content_type) -> pd.DataFrame:
    """Parse input data (csv/json) and return a DataFrame."""

def output_fn(output_df, accept_type):
    """Convert DataFrame to CSV or JSON output formats."""

def predict_fn(df, models) -> pd.DataFrame:
    """Make Predictions with our Model(s) and return a Dataframe"""
```



1. **Prediction Intervals using Bootstrapping**

The Model Script for this **ensemble** model is here:

  
1. **Quantile Regression**
   - A set of models with different objective functions that can provide quantile estimates for predictions, offering a broader sense of the distribution within the training data. It will help in understanding the range and variability of predictions. We'll estimate a range of quantiles that should give us a 'spread' of target values within that region of feature space.

1. **K-Nearest Neighbors (KNN) Model**: This model will use the distances to the nearest neighbors in the feature space. Observations that have close neighbors with low variance in their target values will have higher confidence values, while those in high variance neighborhoods or sparse regions will have lower confidence.



1. **Model Training**: Will use the AWS SageMaker model training functionality to train our two data quality models.
   - **Quantile Regression + Residuals**: Model to predict different quantiles of the target variable. This model will also compute residuals as described above.
   - **K-Nearest Neighbors (KNN) Model**: Develop a KNN-based model to estimate confidence levels based on the density and variability of observations in the feature space.

1. **Endpoint Deployment**: Deploy the models on AWS SageMaker, setting up serverless endpoints for predictions and confidence estimates.
1. **Model Evaluation**: Evaluate the models using the defined metrics, comparing them against the benchmark model.
1. **Visualization and Reporting**: Create visualizations to illustrate the distribution of predictions and confidence levels, and compile a comprehensive report documenting the findings and insights from the project.

By following this workflow, the project aims to deliver robust data quality/confidence models that enhance the interpretability and reliability of machine learning predictions in AWS.



# References
1. Sorkun, M. C., Khetan, A., & Er, S. (2019). *AqSolDB: A curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds*. Scientific Data, 6, 143. [https://doi.org/10.1038/s41597-019-0151-1](https://doi.org/10.1038/s41597-019-0151-1). Dataset: [https://doi.org/10.7910/DVN/OVHAW8](https://doi.org/10.7910/DVN/OVHAW8)

2. Wang, Z., Zhang, Y., & Xu, J. (2024). *Activity Cliff-Informed Contrastive Learning for Molecular Property Prediction*. PLoS Computational Biology. [PMC11643338](https://pmc.ncbi.nlm.nih.gov/articles/PMC11643338)

3. Stumpfe, D., & Bajorath, J. (2019). *Evolving Concept of Activity Cliffs*. ACS Omega, 4(1), 14360–14368. [https://doi.org/10.1021/acsomega.9b02221](https://pubs.acs.org/doi/10.1021/acsomega.9b02221)

4. Mayr, A., Klambauer, G., & Hochreiter, S. (2023). *Exploring QSAR Models for Activity-Cliff Prediction*. Journal of Cheminformatics, 15(1), 1–14. [https://doi.org/10.1186/s13321-023-00708-w](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00708-w)

5. McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426. [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426). GitHub: https://github.com/lmcinnes/umap

6. Plotly Technologies Inc. (2015). *Collaborative data science with Plotly*. Plotly Technologies Inc. [https://plot.ly](https://plot.ly). GitHub: https://github.com/plotly/plotly.py

7. Plotly Technologies Inc. (2017). *Dash: Analytical Web Apps for Python, R, Julia, and Jupyter*. Plotly Technologies Inc. [https://dash.plotly.com](https://dash.plotly.com). GitHub: https://github.com/plotly/dash