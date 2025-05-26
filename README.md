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



### Prediction Intervals using Bootstrapping

The full Model Script for this **ensemble** model is here: [ensemble_bootstrap](https://github.com/brifordwylie/aws_ml_engineer_capstone/blob/main/model_scripts/ensemble_xgb/ensemble_xgb.py). The model script follows many of the general entry points above. Here are some of the relevant details for the creation of 10 'bootstrapped' models. The challenge here was obviously training, saving, loading, and predicting against 10 different models. 

**Training**

```
# Train 10 models with random 80/20 splits of the data
for model_id in range(10):
    # Model Name
    model_name = f"m_{model_id:02}"

    # Randomly split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Training Model {model_name} with {len(X_train)} rows")
    params = {"objective": "reg:squarederror"}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Store the model
    models[model_name] = model
```

**Inference/Loading Models**

```
# Load ALL the models from the model directory
models = {}
for file in os.listdir(model_dir):
    if file.startswith("m_") and file.endswith(".json"):  # The Quantile models
        # Load the model
        model_path = os.path.join(model_dir, file)
        print(f"Loading model: {model_path}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)

        # Store the model
        m_name = os.path.splitext(file)[0]
        models[m_name] = model
```

**Inference/Prediction**

Since we loaded the model (above) into a dictionary of models it made the predictions fairly straight forward.

```
# Predict the features against all the models
for name, model in models.items():
    df[name] = model.predict(matched_df[model_features])
```

**Outputs**

The outputs from the Endpoint has the output of all 10 models. These output can be used to compute a prediction interval for each prediction.

```
        id      m_00      m_01      m_02      m_03      m_04      m_05      m_06      m_07      m_08      m_09  solubility  residuals
0    H-450 -7.016055 -7.259755 -7.145797 -6.947074 -6.958017 -7.003732 -6.958448 -7.012415 -7.026974 -6.952472       -6.94   0.184727
1   B-3169 -7.016055  -7.03502 -6.979581 -6.905605 -7.107079 -7.003732 -6.958448 -6.890439 -7.003515 -6.937048     -6.9069   -0.01326
2   B-4093  -7.12962 -7.213452 -7.231046 -7.310788  -7.13347  -7.15819  -7.09764 -7.069907  -7.40312 -7.103029     -7.7699  -0.626903
3   B-2885 -7.016055  -7.03502 -6.979581 -7.057226 -7.031295 -7.003732 -6.958448 -6.890439 -6.946234 -6.937048     -7.0199  -0.106015
4   B-4094  -7.12962 -7.213452 -7.128245 -7.474997 -7.002088  -7.15819  -7.09764 -7.069907 -7.213766 -7.103029     -6.8754   0.267597
5    C-718 -6.914408  -7.03502 -7.048625 -7.057226 -7.050063 -7.003732 -6.958448 -6.890439 -6.946234  -7.00981       -7.26  -0.290513
6   B-3202 -6.892575  -7.03502 -7.048625 -7.057226 -7.002088 -7.003732 -6.958448 -6.890439 -6.969693  -7.00981       -6.77   0.127826
7   B-4092 -6.801861 -7.213452 -7.048625 -7.286277 -7.020856 -7.003732 -7.017329 -7.061284 -6.969693 -7.088557     -6.8526   0.045226
8   B-3191 -6.914408  -7.03502 -7.048625 -7.057226 -7.020856 -7.003732 -6.958448 -6.890439 -6.987513  -7.00981     -7.1299  -0.160413
9   B-2811 -6.914408  -7.03502 -7.048625 -7.057226 -7.050063 -7.003732 -6.958448 -6.890439 -6.946234  -7.00981     -6.6813   0.288187
10  B-3201 -6.914408  -7.03502 -7.048625 -7.057226 -7.050063 -7.003732 -6.958448 -6.890439 -6.946234  -7.00981       -6.77   0.199487
```


### Quantile Regressor

The full Model Script for this **quantile regressor** model is here: [quantile_regressor](https://github.com/brifordwylie/aws_ml_engineer_capstone/blob/main/model_scripts/quant_regression/quant_regression.py). Here are some of the relevant details for the creation of quantile regressor models. Specifically we use the **reg:quantileerrro** objective function, we create 5 seperate model for the following quantiles **[0.10, 0.25, 0.50, 0.75, 0.90]**

```
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
q_models = {}
...

# Train models for each of the quantiles
for q in quantiles:
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": q,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    # Convert quantile to string
    q_str = f"q_{int(q * 100):02}"

    # Store the model
    q_models[q_str] = model
```

We follow a similar pattern where we save the model and then load them in during endpoint inference and predict against all the models.

**Outputs**
The outputs from this model include the following quantiles **[0.10, 0.25, 0.50, 0.75, 0.90]**. The output also include **IQR** Inner Quartile Range (q_75-q_25) and **IDR** which is the Inner Decile Range (the difference between q_10 and q_90), it gives you an estimate of the entire span of all the target values in that region.

```
[●●●]Workbench:scp_sandbox> quant_df[quant_show]
Out[67]:
        id      q_10      q_25      q_50      q_75      q_90       iqr       idr  prediction  solubility  residuals
0   A-2232  -7.64602 -5.080912 -4.711096 -3.999644 -3.357492  1.081267  4.288528   -5.801939   -6.680445  -0.878507
1    A-690 -6.745965 -4.733729 -4.453765 -3.881248 -3.227552  0.852481  3.518413   -5.084425   -4.701186   0.383239
2    C-987 -5.953816 -4.757437 -4.470902 -3.606793 -2.970807  1.150644  2.983008   -4.617952       -4.74  -0.122048
3   C-2449  -6.34855 -5.794905 -5.082012 -4.740721 -3.539795  1.054184  2.808755   -4.852782       -5.84  -0.987218
4    F-999 -6.398141 -4.438383 -4.733964 -3.994623 -3.526953   0.44376  2.871188    -4.61281       -3.96    0.65281
5    B-873 -7.745192 -6.920116 -6.379264 -5.322877 -3.990977  1.597239  3.754215   -6.401119     -6.5727  -0.171581
6   B-2020 -6.858561 -6.204701 -5.676304 -4.276959  -4.14386  1.927742  2.714701   -4.853285     -3.5738   1.279485
7   C-2396  -6.43478 -5.252554 -5.186502 -4.007359 -3.740023  1.245195  2.694756   -4.748798       -3.85   0.898798
8   C-2463  -6.56106 -5.323305 -5.231742 -4.433026 -3.764848  0.890279  2.796212   -4.514942       -4.13   0.384942
9    F-838 -5.965605 -5.662756 -5.592986 -4.487374 -3.785937  1.175383  2.179669   -5.646573       -5.09   0.556573
10  B-2235 -5.625131 -4.662147 -3.869359 -3.541233  -3.22161  1.120913  2.403521   -3.662866     -3.9031  -0.240234
11  C-1012 -6.029509  -4.84484 -4.574955 -3.780015  -3.05753  1.064825  2.971979   -4.493772        -3.6   0.893772
12  C-2350 -5.809307 -5.230641 -4.826998 -4.205239 -3.261495  1.025402  2.547811   -4.513793       -3.76   0.753793
13   B-872 -7.745192 -6.920116 -6.379264 -5.322877 -3.990977  1.597239  3.754215   -6.401119     -5.8957   0.505419
14  C-1018 -6.356013 -5.332912 -4.772804 -4.077983 -3.549013  1.254929  2.807001   -4.888631       -5.21  -0.321369
15  B-1540 -6.967558 -6.360556 -5.682418 -4.648727 -3.865636  1.711829  3.101922   -6.000487     -7.1031  -1.102613
16   C-948   -5.6817  -4.01971 -3.827436 -3.338722  -2.42288  0.680988  3.258819   -3.872794        -4.0  -0.127206
17  C-1037 -6.372856 -5.500494 -4.997159 -4.657032 -3.759637  0.843461  2.613219   -4.672111       -4.08   0.592111
18   A-886 -6.221915 -5.360593 -4.667631 -4.204701 -3.116444  1.155892  3.105471    -4.69916   -4.891656  -0.192496
19  A-3067 -6.712882 -5.567425 -4.927672 -3.979314 -3.131096  1.588111  3.581786   -4.856071   -4.867097  -0.011026
```


  
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