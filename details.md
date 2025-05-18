# Exploration of quantiles, ensembles, and proximity models
The intent here is to explore a variety of approaches for estimating prediction intervals. These approaches provide a more nuanced view of model uncertainty than simple point predictions, allowing for more informed decision-making in real-world applications.
 

## Expectations
When implementing confidence estimation for regression problems, we expect different behavior in regions of high and low variance. Understanding these behaviors is crucial for interpreting the uncertainty measures provided by our models.

**Areas of low variance**
The ensemble model (10 models), should give fairly consistent predictions even though 20% of the data is randomly dropped out. If we're looking at an area of feature space where all the target values are around 4.0 then each of the 10 models should estimate 4.0 with relatively low variance. This convergence indicates high confidence in predictions.

The quantile model (q\_10, q\_25, q\_50, q\_75, q\_90) should also show tight bounds in these regions. We expect q\_10 and q\_90 to be relatively close to q\_50, reflecting the model's confidence in its prediction. The narrower the interval between quantiles, the higher our confidence.

**Areas of high variance**
In regions of feature space with higher variability in target values, we expect ensemble models to diverge in their predictions. This divergence serves as a natural indicator of uncertainty - when models trained on different subsets of data disagree significantly, we should have lower confidence in any single prediction.

For quantile regression in high-variance areas, we expect to see wider intervals between quantiles (particularly q\_10 to q\_90), reflecting increased uncertainty. The median prediction (q_50) becomes less reliable as a point estimate, making the full distribution more valuable for decision-making.


# Areas of Low Variance
We expect that our models show low variance. The ensemble model should not be that different across the 10 models and the quantile model should show reasonbly tight bounds.

```
low_ids = ['B-3202', 'B-4094', 'B-3169', 'B-3191', 'B-4092', 'B-4093', 
           'B-2885', 'B-3201', 'C-718', 'H-450', 'B-2811']
```

### Ensemble Model
**Bootstrap Aggregating (Bagging)**
The ensemble model uses a traditional bagging approach to train 10 models with random 20% data drop out for each model.

```
[●●●]Workbench:scp_sandbox> ensem_df[ensem_show_details]

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
```
[●●●]Workbench:scp_sandbox> ensem_df[ensem_show]
Out[88]:
        id     p_min     p_max   p_delta  p_stddev  prediction  solubility  residuals
0    H-450 -7.259755 -6.947074  0.312682  0.100306   -7.124727       -6.94   0.184727
1   B-3169 -7.107079 -6.890439   0.21664  0.064398   -6.893641     -6.9069   -0.01326
2   B-4093  -7.40312 -7.069907  0.333213  0.105603   -7.142997     -7.7699  -0.626903
3   B-2885 -7.057226 -6.890439  0.166787  0.052338   -6.913885     -7.0199  -0.106015
4   B-4094 -7.474997 -7.002088   0.47291  0.127839   -7.142997     -6.8754   0.267597
5    C-718 -7.057226 -6.890439  0.166787  0.060268   -6.969487       -7.26  -0.290513
6   B-3202 -7.057226 -6.890439  0.166787  0.059083   -6.897826       -6.77   0.127826
7   B-4092 -7.286277 -6.801861  0.484416  0.131751   -6.897826     -6.8526   0.045226
8   B-3191 -7.057226 -6.890439  0.166787  0.055853   -6.969487     -7.1299  -0.160413
9   B-2811 -7.057226 -6.890439  0.166787  0.060268   -6.969487     -6.6813   0.288187
10  B-3201 -7.057226 -6.890439  0.166787  0.060268   -6.969487       -6.77   0.199487
```


# Areas of Uncertainty

```
mixed_ids = ['A-2232', 'A-3067', 'A-690', 'A-886', 'B-1540', 'B-2020', 'B-2235', 'B-872',
             'B-873', 'C-1012', 'C-1018', 'C-1037', 'C-2350', 'C-2396', 'C-2449', 'C-2463',
             'C-948', 'C-987', 'F-838', 'F-999']
```



### Ensemble Model
**Bootstrap Aggregating (Bagging)**
The ensemble model uses a traditional bagging approach to train 10 models with random 20% data drop out for each model.

```
[●●●]Workbench:scp_sandbox> ensem_df[ensem_show_details]
Out[71]:
        id      m_00      m_01      m_02      m_03      m_04      m_05      m_06      m_07      m_08      m_09  solubility  residuals
0   A-2232 -6.044363 -5.801979 -6.199892 -3.936921 -5.870393 -4.223893  -5.52595 -6.328047 -4.493309 -4.640964   -6.680445  -0.878507
1    A-690 -4.760173 -5.355776  -4.93499 -5.177505 -5.240327  -4.58196 -5.589244 -5.337272 -5.804517 -5.222544   -4.701186   0.383239
2    C-987 -4.341814 -4.517933 -4.616502 -4.655886 -4.464478 -4.174451 -4.354518 -4.572282 -4.641025 -4.574629       -4.74  -0.122048
3   C-2449  -4.98431 -4.383635 -4.855946 -4.975514 -5.249914 -4.736091 -5.108003 -4.604824 -4.098117 -4.631466       -5.84  -0.987218
4    F-999  -4.11748 -4.477765 -4.332422 -3.815905 -4.281518 -4.151437 -4.262539 -3.673705   -4.4757 -3.913755       -3.96    0.65281
5    B-873 -6.254342 -6.000369 -6.022392 -6.141455 -6.382396 -5.944615 -5.985099 -6.275522 -6.422005 -6.867336     -6.5727  -0.171581
6   B-2020 -4.873385  -6.20369 -4.301351 -4.790401 -5.057366 -4.559771 -4.281707 -5.066243 -4.221696 -4.695738     -3.5738   1.279485
7   C-2396 -4.720499 -4.561836 -4.581423 -5.797516 -4.527462 -4.825343 -4.216729 -4.730138 -4.703294 -4.762237       -3.85   0.898798
8   C-2463 -6.093319 -4.511602 -4.668102 -4.312883 -4.563525 -4.455013 -4.515947  -4.21011 -5.129335 -4.835496       -4.13   0.384942
9    F-838 -5.769777  -5.30483 -5.323261 -5.262569 -5.154504 -5.536853 -5.440283 -5.511413 -5.716299 -5.554013       -5.09   0.556573
10  B-2235 -4.003233   -3.8019 -4.183228 -4.032767 -3.887784 -3.800082 -3.929849 -3.701372 -3.647951 -3.681801     -3.9031  -0.240234
11  C-1012 -4.285204 -4.764254 -4.147794 -4.020972 -4.434122 -4.117913 -4.691846 -4.159267 -4.388888 -5.354635        -3.6   0.893772
12  C-2350 -4.133137 -3.998734 -4.419365 -4.082594 -5.152583 -4.256238 -4.193721 -4.493607 -4.177179  -4.05379       -3.76   0.753793
13   B-872 -6.254342 -6.000369 -6.022392 -6.141455 -6.382396 -5.944615 -5.985099 -6.275522 -6.422005 -6.867336     -5.8957   0.505419
14  C-1018 -4.090982 -5.234408 -5.060978 -4.238099 -4.833651 -4.958716 -4.951783 -4.827072 -4.731182 -5.063056       -5.21  -0.321369
15  B-1540 -6.404103 -6.376208 -6.181894   -6.5244 -6.373813 -5.391015 -6.349219 -6.329221  -6.10229 -6.538403     -7.1031  -1.102613
16   C-948 -4.193171 -4.280911 -4.162027 -4.182942 -4.066881 -3.956147 -4.268219 -3.974617 -4.064903 -4.354257        -4.0  -0.127206
17  C-1037 -4.742284 -4.278191 -4.417413  -5.27635 -5.176488  -4.56231 -4.594252 -4.560282 -3.900867 -4.909844       -4.08   0.592111
18   A-886 -5.113581 -5.192486 -4.923438 -4.795242 -4.915673 -4.719914 -5.267296 -4.541485 -4.706693 -5.002457   -4.891656  -0.192496
19  A-3067  -4.61397 -4.369001 -4.711539     -4.84 -4.729492 -4.684245 -4.888774 -5.035903 -4.874368 -4.784814   -4.867097  -0.011026
```
### Ensemble Results
We also have columns for:

- **p_min:** minimum prediction of all models
- **p_max:** maximum prediction of all models
- **p_delta:** the max-min prediciton delta
- **p_stddev:** the standard deviation of the 10 model predictions



```
ensem_df[ensem_show]

        id     p_min     p_max   p_delta  p_stddev  prediction  solubility  residuals
0   A-2232 -6.328047 -3.936921  2.391126  0.891303   -5.801939   -6.680445  -0.878507
1    A-690 -5.804517  -4.58196  1.222557  0.366417   -5.084425   -4.701186   0.383239
2    C-987 -4.655886 -4.174451  0.481435  0.156901   -4.617952       -4.74  -0.122048
3   C-2449 -5.249914 -4.098117  1.151797  0.347679   -4.852782       -5.84  -0.987218
4    F-999 -4.477765 -3.673705   0.80406  0.273272    -4.61281       -3.96    0.65281
5    B-873 -6.867336 -5.944615  0.922721   0.28167   -6.401119     -6.5727  -0.171581
6   B-2020  -6.20369 -4.221696  1.981994  0.580515   -4.853285     -3.5738   1.279485
7   C-2396 -5.797516 -4.216729  1.580787  0.408758   -4.748798       -3.85   0.898798
8   C-2463 -6.093319  -4.21011   1.88321  0.545098   -4.514942       -4.13   0.384942
9    F-838 -5.769777 -5.154504  0.615273   0.19845   -5.646573       -5.09   0.556573
10  B-2235 -4.183228 -3.647951  0.535277  0.172836   -3.662866     -3.9031  -0.240234
11  C-1012 -5.354635 -4.020972  1.333663   0.40485   -4.493772        -3.6   0.893772
12  C-2350 -5.152583 -3.998734  1.153849  0.338832   -4.513793       -3.76   0.753793
13   B-872 -6.867336 -5.944615  0.922721   0.28167   -6.401119     -5.8957   0.505419
14  C-1018 -5.234408 -4.090982  1.143426  0.365038   -4.888631       -5.21  -0.321369
15  B-1540 -6.538403 -5.391015  1.147388  0.332234   -6.000487     -7.1031  -1.102613
16   C-948 -4.354257 -3.956147   0.39811  0.132671   -3.872794        -4.0  -0.127206
17  C-1037  -5.27635 -3.900867  1.375482   0.41059   -4.672111       -4.08   0.592111
18   A-886 -5.267296 -4.541485  0.725811  0.231779    -4.69916   -4.891656  -0.192496
19  A-3067 -5.035903 -4.369001  0.666902  0.181219   -4.856071   -4.867097  -0.011026
```

#### Let's look at the feature space neighbors of `A-2232`

```
prox_end.inference(df[df["id"]=="A-2232"])

        id neighbor_id  distance  solubility solubility_class
0   A-2232      A-2232       0.0   -6.680445              low
1   A-2232       F-838  0.570097       -5.09              low
2   A-2232      C-2463  0.577817       -4.13           medium
3   A-2232      C-1012  0.594608        -3.6             high
4   A-2232      C-1018  0.627197       -5.21              low
5   A-2232       A-690  0.717372   -4.701186           medium
6   A-2232      C-1037  0.741838       -4.08           medium
7   A-2232      C-2350  0.800967       -3.76             high
8   A-2232       C-948  0.821772        -4.0           medium
9   A-2232       F-999  0.839041       -3.96             high
10  A-2232      B-1540  0.882311     -7.1031              low
```

**How does this compare with our ensemble output?**

```
# Summary
    id     p_min     p_max   p_delta  p_stddev  prediction  solubility  residuals
A-2232 -6.328047 -3.936921  2.391126  0.891303   -5.801939   -6.680445  -0.878507

# Details
    id      m_00      m_01      m_02      m_03      m_04      m_05      m_06      m_07      m_08      m_09  solubility  residuals
A-2232 -6.044363 -5.801979 -6.199892 -3.936921 -5.870393 -4.223893  -5.52595 -6.328047 -4.493309 -4.640964   -6.680445  -0.878507
```



### Quantile Results

```
quant_df[quant_show]

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

### What is IDR?
So IQR is the Inner Quartile Range and IDR is the Inner Decile Range (the difference between q_10 and q\_90), it gives you an estimate of the entire span of all the target values in that region.

#### Let's look at a high IDR neighborhood
```
   id      q_10      q_25      q_50      q_75      q_90       iqr       idr  prediction  solubility  residuals
B-873 -7.745192 -6.920116 -6.379264 -5.322877 -3.990977  1.597239  3.754215   -6.401119     -6.5727  -0.171581
```

```
prox_end.inference(df[df["id"]=="B-873"])

       id neighbor_id  distance  solubility solubility_class
0   B-873       B-872       0.0     -5.8957              low
1   B-873       B-873       0.0     -6.5727              low
2   B-873      B-2235  0.744616     -3.9031             high
3   B-873      B-1540  0.942533     -7.1031              low
4   B-873      C-2396  1.013237       -3.85             high
5   B-873      C-2215  1.031921       -6.75              low
6   B-873      C-1478  1.040356       -4.86           medium
7   B-873      B-2192  1.040356     -4.8638           medium
8   B-873      B-3585  1.054484      -7.733              low
9   B-873      B-2738  1.057538      -6.678              low

```

#### Let's look at a low (relative) IDR neighborhood
```
    id      q_10      q_25      q_50      q_75      q_90       iqr       idr  prediction  solubility  residuals
B-2235 -5.625131 -4.662147 -3.869359 -3.541233  -3.22161  1.120913  2.403521   -3.662866     -3.9031  -0.240234
```
```
prox_end.inference(df[df["id"]=="B-2235"])

        id neighbor_id  distance  solubility solubility_class
0   B-2235      B-2235       0.0     -3.9031             high
1   B-2235      B-2192  0.620994     -4.8638           medium
2   B-2235      C-1478  0.621075       -4.86           medium
3   B-2235      C-2396  0.705833       -3.85             high
4   B-2235       B-872  0.744616     -5.8957              low
5   B-2235       B-873  0.744616     -6.5727              low
6   B-2235       A-690  0.951497   -4.701186           medium
7   B-2235       F-838  0.958551       -5.09              low
8   B-2235       A-886  0.980354   -4.891656           medium
9   B-2235       C-987  0.989415       -4.74           medium
```

## Odd man out
image here

```
prox_end.inference(df[df["id"]=="A-5756"])

        id neighbor_id  distance  solubility solubility_class
0   A-5756      A-5756       0.0   -7.686268              low
1   A-5756      B-1720  0.671356      -0.281             high
2   A-5756      B-1406  0.718416     -1.5044             high
3   A-5756      A-5392  0.723558    -0.03759             high
4   A-5756      B-1711   0.80598     -1.4593             high
5   A-5756       G-875  0.810128       -0.22             high
6   A-5756       E-602  0.826331        -0.8             high
7   A-5756      B-4265  0.838613      1.1303             high
8   A-5756      A-1720  0.849515     0.18526             high
9   A-5756      A-4570  0.869663   -1.025801             high
10  A-5756      A-3612  0.872098   -0.372768             high
```

### Ensemble Results
```
[●●●]Workbench:scp_sandbox> ensem_df[ensem_show_details]
Out[99]:
        id      m_00      m_01      m_02      m_03      m_04      m_05      m_06      m_07      m_08      m_09  solubility  residuals
0    E-602 -0.628571 -0.452322 -0.282568 -0.419866 -0.586266 -0.486029 -0.466363 -0.588855 -0.667487 -0.504331        -0.8  -0.386523
1    G-875 -0.097696   0.05653  0.219278  0.123607   0.15991  0.254011 -0.178735  0.147489  0.072352   -0.0748       -0.22  -0.323315
2   A-5756 -4.138385 -7.008507 -6.794707 -7.305354 -7.027105 -6.648158 -6.655486 -6.877338 -6.714417 -7.112392   -7.686268  -1.211766
3   A-5392 -0.379488 -0.516007 -0.381355  -0.53043 -0.528278 -0.745296 -0.422248 -0.239877 -0.419536 -0.479364    -0.03759   0.427638
4   B-1720 -0.185579  -0.25287 -0.243852 -0.366501 -0.381324 -0.206033 -0.409947 -0.128945 -0.447178 -0.312244      -0.281  -0.003135
5   A-3612  0.031073  0.092471 -0.169421  0.246318 -0.088565 -0.248758  0.058354 -0.176708  0.170942  -0.08127   -0.372768  -0.383087
6   B-1711 -1.173072 -1.057656  -1.10061 -0.885798  -1.33391 -1.013229 -1.441702 -0.810729 -0.822198 -0.764714     -1.4593  -0.612358
7   B-4265  0.763196  0.337098  0.597087  0.738875   0.66697  0.100463  0.914631  0.641585  0.477614  0.397362      1.1303   0.352136
8   B-1406 -1.136586 -0.963558 -1.080088 -1.033957 -1.080255 -0.983202 -1.373497 -0.962505 -0.769203 -1.047441     -1.5044  -0.749002
9   A-1720  0.254489   0.26452  0.465777  0.373203  0.257408  0.201058  0.362782  0.288272  0.491525  0.096377     0.18526  -0.030979
10  A-4570 -0.028044 -0.072333  0.082238 -0.018812 -0.697143  0.205901 -0.086977  -0.22728 -0.389696 -0.183299   -1.025801  -0.782443

        id     p_min     p_max   p_delta  p_stddev  prediction  solubility  residuals
0    E-602 -0.667487 -0.282568  0.384919  0.113799   -0.413477        -0.8  -0.386523
1    G-875 -0.178735  0.254011  0.432745  0.143069    0.103315       -0.22  -0.323315
2   A-5756 -7.305354 -4.138385  3.166969  0.900265   -6.474502   -7.686268  -1.211766
3   A-5392 -0.745296 -0.239877  0.505419  0.132722   -0.465228    -0.03759   0.427638
4   B-1720 -0.447178 -0.128945  0.318233  0.105967   -0.277865      -0.281  -0.003135
5   A-3612 -0.248758  0.246318  0.495077  0.162047    0.010319   -0.372768  -0.383087
6   B-1711 -1.441702 -0.764714  0.676988  0.228454   -0.846942     -1.4593  -0.612358
7   B-4265  0.100463  0.914631  0.814168   0.23863    0.778164      1.1303   0.352136
8   B-1406 -1.373497 -0.769203  0.604293  0.153544   -0.755398     -1.5044  -0.749002
9   A-1720  0.096377  0.491525  0.395147  0.120042     0.21624     0.18526  -0.030979
10  A-4570 -0.697143  0.205901  0.903045  0.255141   -0.243358   -1.025801  -0.782443
```

### Quantile Results
```
[●●●]Workbench:scp_sandbox> quant_df[quant_show]
Out[104]:
        id      q_10      q_25      q_50      q_75      q_90       iqr       idr  prediction  solubility  residuals
0    E-602 -0.850253 -0.868499 -0.253259 -0.275721  0.717461  0.592777  1.567714   -0.413477        -0.8  -0.386523
1    G-875 -0.499022 -0.113913  0.146836  0.206248  0.915378  0.320161  1.414401    0.103315       -0.22  -0.323315
2   A-5756 -7.702628 -6.865536  -3.78436 -1.732224   0.08871  5.133312  7.791338   -6.474502   -7.686268  -1.211766
3   A-5392 -0.702735 -0.521378 -0.114554   0.04623 -0.025863  0.567608  0.676873   -0.465228    -0.03759   0.427638
4   B-1720 -0.499022 -0.312891 -0.387124 -0.129966  0.875888  0.182925   1.37491   -0.277865      -0.281  -0.003135
5   A-3612 -0.785785 -0.485026  0.074099  0.631392  0.862679  1.116418  1.648464    0.010319   -0.372768  -0.383087
6   B-1711 -1.327407   -1.0744 -0.778048 -0.360076  0.634595  0.714324  1.962002   -0.846942     -1.4593  -0.612358
7   B-4265  -0.14556  0.087534  0.660189  0.870833  1.026695  0.783298  1.172255    0.778164      1.1303   0.352136
8   B-1406 -1.618196 -1.089152 -0.758939 -0.261084  0.638641  0.828067  2.256837   -0.755398     -1.5044  -0.749002
9   A-1720 -0.363344 -0.033849  0.452073  0.684097  0.977878  0.717947  1.341222     0.21624     0.18526  -0.030979
10  A-4570 -1.014792 -0.428357  0.249862  0.682359  0.869382  1.110716  1.884174   -0.243358   -1.025801  -0.782443
```

## A-1392

### Neighbors
```
        id neighbor_id  distance  solubility solubility_class
0   A-1392      A-1392       0.0   -2.573579             high
1   A-1392      A-5820  0.028985   -6.575188              low
2   A-1392      A-5686   0.19604   -5.117146              low
3   A-1392      A-2152  0.198466   -5.297938              low
4   A-1392      A-6080  0.198528   -7.968044              low
5   A-1392      A-5086  0.199283   -4.894775           medium
6   A-1392      A-5844  0.199655   -5.519098              low
7   A-1392        A-55  0.204004   -4.203848           medium
8   A-1392      A-5563  0.206527     -10.017              low
9   A-1392      A-3275  0.209462   -0.814425             high
10  A-1392      A-2604   0.21302   -5.358049              low
```

## Top 10 Shapley Values

```
['mollogp',
 'bertzct',
 'molwt',
 'tpsa',
 'numvalenceelectrons',
 'balabanj',
 'molmr',
 'labuteasa',
 'numhdonors',
 'numheteroatoms']
```

# Storage
## Implementation Approaches

### Ensemble Methods
1. **Bootstrap Aggregating (Bagging)**
   - Train multiple models on random subsets of data
   - Use variance in predictions as uncertainty measure
   - Calculate prediction intervals as percentiles of ensemble outputs

2. **Dropout Ensemble**
   - Apply random feature/data dropout during training
   - Generate multiple predictions with different dropout patterns
   - Interpret variance as epistemic uncertainty

### Quantile Regression
1. **Direct Quantile Prediction**
   - Train models to directly predict specific quantiles (10th, 25th, 50th, 75th, 90th)
   - Use asymmetric loss functions (pinball loss)
   - Prediction interval = range between lower and upper quantiles

2. **Quantile Forests**
   - Leverage random forests for non-parametric quantile estimation
   - Naturally handles heteroscedastic data
   - Provides full conditional distribution

### Proximity Models
1. **Nearest Neighbor Variance**
   - Examine variance of target values in local neighborhood
   - Higher local variance indicates higher uncertainty
   - Calculate confidence as inverse of local variance

2. **Distance-Weighted Uncertainty**
   - Weight uncertainty by distance to training examples
   - Greater distance to training data implies higher uncertainty
   - Combine with other methods for robust intervals

## Evaluation Metrics

1. **Prediction Interval Coverage Probability (PICP)**
   - Percentage of true values falling within prediction interval
   - PICP should match nominal confidence level (e.g., 80% for q_10-q_90)

2. **Mean Prediction Interval Width (MPIW)**
   - Average width of prediction intervals
   - Balance against PICP (narrower intervals with same coverage are better)

3. **Continuous Ranked Probability Score (CRPS)**
   - Proper scoring rule for probabilistic forecasts
   - Measures entire predictive distribution quality

## Integration with Production Pipeline

When integrating these methods into production:

1. **Serving Multiple Outputs**
   - Return point prediction, intervals, and confidence metric
   - Enable downstream processes to make appropriate decisions

2. **Thresholding**
   - Set confidence thresholds for automated vs. human review
   - Adjust thresholds based on application requirements

3. **Feedback Loop**
   - Monitor actual outcomes vs. prediction intervals
   - Recalibrate uncertainty estimates as needed

4. **Visualization**
   - Present uncertainty visually to end-users
   - Make uncertainty an explicit part of the decision process