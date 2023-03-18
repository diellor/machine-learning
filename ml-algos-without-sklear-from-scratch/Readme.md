### Below you will find information on how the assignments were give, and in this directory, you will find all the solutions for each of the tasks below.

### Assignment: linear_regression_manual
Solve a linear regression problem using the algorithm from the lecture
which explicitly computes the matrix inversion. Then compute root mean square
error on the test set.

#### Tests Start: linear_regression_manual_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 linear_regression_manual.py --test_size=0.1`
```
52.38
```
- `python3 linear_regression_manual.py --test_size=0.5`
```
54.58
```
- `python3 linear_regression_manual.py --test_size=0.9`
```
59.46
```
#### Tests End:
```
### Assignment: linear_regression_features
Try using a concatenation of features $x^1, x^2, …, x^D$ for $D$ from 1 to
a given range, and report RMSE of every such configuration.

#### Tests Start: linear_regression_features_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 linear_regression_features.py --data_size=10 --test_size=5 --range=6`
```
Maximum feature order 1: 0.74 RMSE
Maximum feature order 2: 1.87 RMSE
Maximum feature order 3: 0.53 RMSE
Maximum feature order 4: 4.52 RMSE
Maximum feature order 5: 1.70 RMSE
Maximum feature order 6: 2.82 RMSE
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_features_1.svgz)
- `python3 linear_regression_features.py --data_size=30 --test_size=20 --range=9`
```
Maximum feature order 1: 0.56 RMSE
Maximum feature order 2: 1.53 RMSE
Maximum feature order 3: 1.10 RMSE
Maximum feature order 4: 0.28 RMSE
Maximum feature order 5: 1.60 RMSE
Maximum feature order 6: 3.09 RMSE
Maximum feature order 7: 3.92 RMSE
Maximum feature order 8: 65.11 RMSE
Maximum feature order 9: 3886.97 RMSE
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_features_2.svgz)
- `python3 linear_regression_features.py --data_size=50 --test_size=40 --range=9`
```
Maximum feature order 1: 0.63 RMSE
Maximum feature order 2: 0.73 RMSE
Maximum feature order 3: 0.31 RMSE
Maximum feature order 4: 0.26 RMSE
Maximum feature order 5: 1.22 RMSE
Maximum feature order 6: 0.69 RMSE
Maximum feature order 7: 2.39 RMSE
Maximum feature order 8: 7.28 RMSE
Maximum feature order 9: 201.70 RMSE
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_features_3.svgz)
#### Tests End:
```

### Assignment: rental_competition
#### Date: Deadline: Oct 24, 7:59 a.m.
#### Points: 3 points+4 bonus

This assignment is a [competition task](https://ufal.mff.cuni.cz/courses/npfl129/2223-winter#competitions).
Your goal is to perform regression on the data from a bike rental shop.
The train set contains 1000 instances, each instance consists of 12 features,
both integral and real.

The [rental_competition.py](https://github.com/ufal/npfl129/tree/master/labs/02/rental_competition.py)
template shows how to load the training data, downloading it if needed.
Furthermore, it shows how to save a trained estimator and how to load it during
prediction.

The performance of your system is measured using _root mean squared error_
and your goal is to achieve RMSE less than 100. Note that you can use
any number of **generalized linear models from sklearn** to solve
this assignment (but no decision trees, MLPs, …).
```


### Assignment: feature_engineering
#### Date: Deadline: Oct 24, 7:59 a.m.
#### Points: 3 points
#### Tests: feature_engineering_tests

Starting with the [feature_engineering.py](https://github.com/ufal/npfl129/tree/master/labs/02/feature_engineering.py)
template, learn how to perform basic feature engineering using `scikit-learn`.

#### Tests Start: feature_engineering_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 feature_engineering.py --dataset=diabetes`
```
-0.5745 -0.9514 1.797 -0.4984 0.4751 0.9487 -0.6961 0.7574 0.06019 1.625 0.33 0.5465 -1.033 0.2863 -0.2729 -0.545 0.3999 -0.4351 -0.03458 -0.9334 0.9052 -1.71 0.4742 -0.452 -0.9026 0.6623 -0.7206 -0.05727 -1.546 3.23 -0.8959 0.8539 1.705 -1.251 1.361 0.1082 2.92 0.2484 -0.2368 -0.4729 0.347 -0.3775 -0.03 -0.8099 0.2257 0.4507 -0.3307 0.3598 0.0286 0.7719 0.9 -0.6604 0.7185 0.0571 1.541 0.4845 -0.5272 -0.0419 -1.131 0.5736 0.04559 1.231 0.003623 0.0978 2.64
0.2776 -0.9514 0.08366 -1.148 -1.592 -1.397 -0.4687 -0.7816 -0.3766 -1.973 0.07706 -0.2641 0.02322 -0.3186 -0.442 -0.3878 -0.1301 -0.217 -0.1045 -0.5477 0.9052 -0.0796 1.092 1.515 1.329 0.4459 0.7436 0.3583 1.877 0.007 -0.09602 -0.1332 -0.1169 -0.03921 -0.06539 -0.03151 -0.1651 1.317 1.827 1.603 0.5379 0.8971 0.4322 2.264 2.535 2.224 0.7462 1.245 0.5996 3.141 1.952 0.6548 1.092 0.5261 2.757 0.2197 0.3663 0.1765 0.9247 0.6109 0.2944 1.542 0.1418 0.7431 3.893
0.8198 1.051 -0.683 -0.8108 -0.6896 -0.4871 -0.2413 -0.03186 -0.2682 0.04527 0.6721 0.8617 -0.5599 -0.6647 -0.5653 -0.3993 -0.1978 -0.02612 -0.2199 0.03711 1.105 -0.7179 -0.8522 -0.7248 -0.512 -0.2536 -0.03348 -0.2819 0.04758 0.4665 0.5538 0.471 0.3327 0.1648 0.02176 0.1832 -0.03092 0.6574 0.5591 0.3949 0.1956 0.02583 0.2175 -0.0367 0.4755 0.3359 0.1664 0.02197 0.185 -0.03121 0.2373 0.1175 0.01552 0.1306 -0.02205 0.05822 0.007686 0.06472 -0.01092 0.001015 0.008544 -0.001442 0.07194 -0.01214 0.002049
0.9747 1.051 1.211 0.6803 0.6207 -0.9859 -1.151 1.547 2.783 2.853 0.9501 1.025 1.18 0.6631 0.605 -0.961 -1.122 1.508 2.712 2.781 1.105 1.273 0.715 0.6524 -1.036 -1.21 1.626 2.925 2.999 1.467 0.8239 0.7517 -1.194 -1.394 1.873 3.37 3.456 0.4628 0.4222 -0.6707 -0.7829 1.052 1.893 1.941 0.3852 -0.6119 -0.7143 0.96 1.727 1.771 0.972 1.135 -1.525 -2.743 -2.813 1.325 -1.78 -3.203 -3.284 2.392 4.304 4.413 7.743 7.94 8.142
-0.1872 -0.9514 0.1739 -1.171 -0.5149 -0.8915 0.5925 -0.8211 0.3554 -0.1302 0.03503 0.1781 -0.03254 0.2193 0.09637 0.1669 -0.1109 0.1537 -0.06651 0.02438 0.9052 -0.1654 1.115 0.4899 0.8482 -0.5637 0.7812 -0.3381 0.1239 0.03023 -0.2037 -0.08952 -0.155 0.103 -0.1428 0.06178 -0.02264 1.372 0.6032 1.044 -0.6941 0.9619 -0.4163 0.1526 0.2651 0.459 -0.3051 0.4228 -0.183 0.06706 0.7948 -0.5282 0.732 -0.3168 0.1161 0.3511 -0.4865 0.2106 -0.07717 0.6742 -0.2918 0.1069 0.1263 -0.04628 0.01696
0.9747 -0.9514 -0.1869 -0.3058 2.659 2.728 0.3651 0.7574 0.676 -0.1302 0.9501 -0.9274 -0.1822 -0.2981 2.592 2.659 0.3559 0.7382 0.6589 -0.1269 0.9052 0.1778 0.291 -2.53 -2.596 -0.3474 -0.7206 -0.6431 0.1239 0.03494 0.05717 -0.497 -0.51 -0.06824 -0.1416 -0.1264 0.02434 0.09354 -0.8132 -0.8344 -0.1117 -0.2316 -0.2067 0.03983 7.07 7.254 0.9708 2.014 1.797 -0.3463 7.443 0.9961 2.066 1.844 -0.3553 0.1333 0.2765 0.2468 -0.04755 0.5736 0.512 -0.09864 0.457 -0.08804 0.01696
1.982 -0.9514 0.715 0.4877 -0.5149 -0.3253 -0.01389 -0.8211 -0.4683 -0.4813 3.927 -1.885 1.417 0.9664 -1.02 -0.6447 -0.02753 -1.627 -0.928 -0.9537 0.9052 -0.6803 -0.464 0.4899 0.3095 0.01322 0.7812 0.4455 0.4579 0.5113 0.3487 -0.3682 -0.2326 -0.009932 -0.5871 -0.3348 -0.3441 0.2378 -0.2511 -0.1586 -0.006774 -0.4004 -0.2284 -0.2347 0.2651 0.1675 0.007152 0.4228 0.2411 0.2478 0.1058 0.004519 0.2671 0.1523 0.1566 0.000193 0.01141 0.006505 0.006685 0.6742 0.3845 0.3952 0.2193 0.2254 0.2316
1.362 1.051 -0.1418 -0.2337 2.193 1.084 1.123 -0.03186 1.76 -0.3935 1.855 1.432 -0.1932 -0.3183 2.987 1.476 1.53 -0.04339 2.397 -0.536 1.105 -0.1491 -0.2456 2.305 1.139 1.18 -0.03348 1.85 -0.4136 0.02011 0.03314 -0.311 -0.1537 -0.1593 0.004518 -0.2496 0.05581 0.05462 -0.5125 -0.2532 -0.2625 0.007445 -0.4113 0.09196 4.809 2.376 2.463 -0.06986 3.86 -0.8629 1.174 1.217 -0.03452 1.907 -0.4264 1.261 -0.03578 1.977 -0.4419 0.001015 -0.05607 0.01253 3.098 -0.6926 0.1548
2.059 -0.9514 1.031 1.69 1.174 0.8206 -1.606 3.046 2.055 1.274 4.24 -1.959 2.122 3.48 2.417 1.69 -3.306 6.273 4.231 2.623 0.9052 -0.9806 -1.608 -1.117 -0.7807 1.528 -2.898 -1.955 -1.212 1.062 1.742 1.21 0.8458 -1.655 3.14 2.118 1.313 2.857 1.984 1.387 -2.714 5.149 3.473 2.153 1.378 0.9633 -1.885 3.576 2.412 1.495 0.6734 -1.318 2.5 1.686 1.045 2.578 -4.891 -3.3 -2.045 9.279 6.26 3.88 4.223 2.618 1.623
0.2776 1.051 -0.48 -0.0173 0.8245 1.178 -0.1655 0.7574 -0.1065 -0.218 0.07706 0.2918 -0.1333 -0.004801 0.2289 0.327 -0.04594 0.2102 -0.02956 -0.06051 1.105 -0.5046 -0.01818 0.8666 1.238 -0.1739 0.7961 -0.1119 -0.2291 0.2304 0.008303 -0.3958 -0.5654 0.07944 -0.3636 0.05111 0.1046 0.0002992 -0.01426 -0.02037 0.002862 -0.0131 0.001842 0.00377 0.6798 0.9711 -0.1364 0.6245 -0.08779 -0.1797 1.387 -0.1949 0.8921 -0.1254 -0.2568 0.02739 -0.1253 0.01762 0.03608 0.5736 -0.08064 -0.1651 0.01134 0.02321 0.04752
```
- `python3 feature_engineering.py --dataset=linnerud`
```
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
```
- `python3 feature_engineering.py --dataset=wine`
```
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.2976 1.31 0.1177 0.9155 0.7783 0.5271 0.4232 0.6668 -1.122 1.048 0.7913 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.282 -1.41 0.8239 -0.4697 -0.1538 0.2001 -1.152 1.377 -0.9146 -0.6609 0.7232 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.442 1.419 0.5492 -1.8 1.03 0.9984 -1.302 0.8977 -0.02794 -0.2336 1.336 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.4121 -0.9989 -1.844 -0.3312 -1.263 -0.6176 -0.6272 -0.3989 -1.174 0.4073 0.301 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.9155 -0.7526 -0.2354 0.8601 -0.7751 -0.3001 0.4232 -0.02594 -1.174 1.646 -0.3936 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.778 0.6071 0.7454 -1.245 0.586 0.9888 -1.527 0.1517 -0.02794 0.06549 1.105 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.43 1.465 0.2746 -0.2204 0.8079 0.6233 -0.5521 -0.5766 0.03261 -0.3191 1.064 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.03446 0.3424 1.295 0.3614 -1.13 -1.445 1.173 -1.465 -0.2442 -0.7464 -0.3255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0.8809 -0.8529 1.295 0.777 1.03 1.2 -0.6272 1.431 0.2316 1.048 0.2193 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.6752 -1.154 -1.765 -0.02646 -0.2869 -0.001945 -0.7772 -0.9496 -0.2096 0.7491 1.268 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
```
#### Tests End:
```

### Assignment: linear_regression_sgd
#### Date: Deadline: Oct 24, 7:59 a.m.
#### Points: 4 points
#### Tests: linear_regression_sgd_tests

Starting with the [linear_regression_sgd.py](https://github.com/ufal/npfl129/tree/master/labs/02/linear_regression_sgd.py),
implement minibatch SGD for linear regression and compare the results to an
explicit linear regression solver.

#### Tests Start: linear_regression_sgd_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 linear_regression_sgd.py --batch_size=10 --epochs=50 --learning_rate=0.01`
```
Test RMSE: SGD 90.96, explicit 91.38
Learned weights: 3.94 7.52 0.08 30.82 -1.72 -1.13 -1.98 6.29 1.98 -10.60 -13.84 -4.31 ...
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_sgd_1.svgz)
- `python3 linear_regression_sgd.py --batch_size=10 --epochs=50 --learning_rate=0.1`
```
Test RMSE: SGD 90.73, explicit 91.38
Learned weights: 1.94 8.31 1.22 33.18 -3.74 -3.64 -2.46 5.19 1.72 -12.40 -14.08 -2.28 ...
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_sgd_2.svgz)
- `python3 linear_regression_sgd.py --batch_size=10 --epochs=50 --learning_rate=0.001`
```
Test RMSE: SGD 108.66, explicit 91.38
Learned weights: 2.79 2.19 -0.06 14.16 -1.07 0.97 0.78 4.62 0.79 -4.62 -7.37 -3.07 ...
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_sgd_3.svgz)
- `python3 linear_regression_sgd.py --batch_size=1  --epochs=50 --learning_rate=0.01`
```
Test RMSE: SGD 90.73, explicit 91.38
Learned weights: 1.94 8.31 1.22 33.18 -3.74 -3.64 -2.46 5.19 1.72 -12.40 -14.08 -2.28 ...
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_sgd_4.svgz)
- `python3 linear_regression_sgd.py --batch_size=50 --epochs=50 --learning_rate=0.01`
```
Test RMSE: SGD 99.74, explicit 91.38
Learned weights: 3.99 3.67 -0.20 20.79 -1.29 1.37 0.47 6.54 1.54 -6.95 -10.65 -4.50 ...
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_sgd_5.svgz)
- `python3 linear_regression_sgd.py --batch_size=50 --epochs=500 --learning_rate=0.01`
```
Test RMSE: SGD 90.67, explicit 91.38
Learned weights: 3.20 8.00 0.57 32.21 -2.49 -2.45 -2.28 5.57 1.69 -11.47 -14.00 -3.44 ...
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_sgd_6.svgz)
- `python3 linear_regression_sgd.py --batch_size=50 --epochs=500 --learning_rate=0.01 --l2=0.1`
```
Test RMSE: SGD 90.71, explicit 91.38
Learned weights: 3.40 7.36 0.32 30.21 -2.14 -1.68 -1.88 5.79 1.72 -10.84 -13.45 -3.73 ...
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_sgd_7.svgz)
#### Tests End:
```

### Assignment: linear_regression_l2
#### Date: Deadline: Oct 24, 7:59 a.m.
#### Points: 2 points
#### Tests: linear_regression_l2_tests

Starting with the [linear_regression_l2.py](https://github.com/ufal/npfl129/tree/master/labs/02/linear_regression_l2.py)
template, use `scikit-learn` to train L2-regularized linear regression models
and print the results of the best of them.

#### Tests Start: linear_regression_l2_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 linear_regression_l2.py --test_size=0.15`
```
0.49 52.11
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_l2_1.svgz)
- `python3 linear_regression_l2.py --test_size=0.80`
```
0.10 53.53
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/linear_regression_l2_2.svgz)
#### Tests End:
```
### Assignment: grid_search
#### Date: Deadline: Oct 31, 7:59 a.m.
#### Points: 2 points
#### Examples: grid_search_examples
#### Tests: grid_search_tests

Starting with [grid_search.py](https://github.com/ufal/npfl129/tree/master/labs/03/grid_search.py)
template, perform a hyperparameter grid search, evaluating hyperparameter performance
using a stratified k-fold crossvalidation, and finally evaluate a model
trained with best hyperparameters on all training data. The easiest way is
to utilize `sklearn.model_selection.GridSearchCV`.

#### Examples Start: grid_search_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 grid_search.py --test_size=0.5`
```
Rank: 11 Cross-val: 86.7% lr__C: 0.01  lr__solver: lbfgs polynomial__degree: 1    
Rank:  5 Cross-val: 92.7% lr__C: 0.01  lr__solver: lbfgs polynomial__degree: 2    
Rank: 11 Cross-val: 86.7% lr__C: 0.01  lr__solver: sag   polynomial__degree: 1    
Rank:  5 Cross-val: 92.7% lr__C: 0.01  lr__solver: sag   polynomial__degree: 2    
Rank:  7 Cross-val: 90.8% lr__C: 1     lr__solver: lbfgs polynomial__degree: 1    
Rank:  3 Cross-val: 96.8% lr__C: 1     lr__solver: lbfgs polynomial__degree: 2    
Rank:  7 Cross-val: 90.8% lr__C: 1     lr__solver: sag   polynomial__degree: 1    
Rank:  4 Cross-val: 96.8% lr__C: 1     lr__solver: sag   polynomial__degree: 2    
Rank: 10 Cross-val: 90.1% lr__C: 100   lr__solver: lbfgs polynomial__degree: 1    
Rank:  1 Cross-val: 97.2% lr__C: 100   lr__solver: lbfgs polynomial__degree: 2    
Rank:  9 Cross-val: 90.5% lr__C: 100   lr__solver: sag   polynomial__degree: 1    
Rank:  2 Cross-val: 97.0% lr__C: 100   lr__solver: sag   polynomial__degree: 2    
Test accuracy: 98.33
```
- `python3 grid_search.py --test_size=0.7`
```
Rank: 11 Cross-val: 87.9% lr__C: 0.01  lr__solver: lbfgs polynomial__degree: 1    
Rank:  5 Cross-val: 91.8% lr__C: 0.01  lr__solver: lbfgs polynomial__degree: 2    
Rank: 11 Cross-val: 87.9% lr__C: 0.01  lr__solver: sag   polynomial__degree: 1    
Rank:  5 Cross-val: 91.8% lr__C: 0.01  lr__solver: sag   polynomial__degree: 2    
Rank:  7 Cross-val: 91.3% lr__C: 1     lr__solver: lbfgs polynomial__degree: 1    
Rank:  3 Cross-val: 95.9% lr__C: 1     lr__solver: lbfgs polynomial__degree: 2    
Rank:  7 Cross-val: 91.3% lr__C: 1     lr__solver: sag   polynomial__degree: 1    
Rank:  4 Cross-val: 95.7% lr__C: 1     lr__solver: sag   polynomial__degree: 2    
Rank: 10 Cross-val: 89.2% lr__C: 100   lr__solver: lbfgs polynomial__degree: 1    
Rank:  1 Cross-val: 96.5% lr__C: 100   lr__solver: lbfgs polynomial__degree: 2    
Rank:  9 Cross-val: 89.2% lr__C: 100   lr__solver: sag   polynomial__degree: 1    
Rank:  2 Cross-val: 96.1% lr__C: 100   lr__solver: sag   polynomial__degree: 2    
Test accuracy: 96.98
```
#### Examples End:
#### Tests Start: grid_search_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 grid_search.py --test_size=0.5`
```
Test accuracy: 98.33
```
- `python3 grid_search.py --test_size=0.7`
```
Test accuracy: 96.98
```
#### Tests End:
```

### Assignment: logistic_regression_sgd
#### Date: Deadline: Oct 31, 7:59 a.m.
#### Points: 5 points
#### Tests: logistic_regression_sgd_tests

Starting with the [logistic_regression_sgd.py](https://github.com/ufal/npfl129/tree/master/labs/03/logistic_regression_sgd.py),
implement minibatch SGD for logistic regression.

#### Tests Start: logistic_regression_sgd_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 logistic_regression_sgd.py --data_size=100 --batch_size=10 --epochs=9 --learning_rate=0.5`
```
After epoch 1: train loss 0.3259 acc 94.0%, test loss 0.3301 acc 96.0%
After epoch 2: train loss 0.2321 acc 96.0%, test loss 0.2385 acc 98.0%
After epoch 3: train loss 0.1877 acc 98.0%, test loss 0.1949 acc 98.0%
After epoch 4: train loss 0.1612 acc 98.0%, test loss 0.1689 acc 98.0%
After epoch 5: train loss 0.1435 acc 98.0%, test loss 0.1517 acc 98.0%
After epoch 6: train loss 0.1307 acc 98.0%, test loss 0.1396 acc 98.0%
After epoch 7: train loss 0.1208 acc 98.0%, test loss 0.1304 acc 96.0%
After epoch 8: train loss 0.1129 acc 98.0%, test loss 0.1230 acc 96.0%
After epoch 9: train loss 0.1065 acc 98.0%, test loss 0.1170 acc 96.0%
Learned weights 2.77 -0.60 0.12
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/logistic_regression_sgd_1.svgz)
- `python3 logistic_regression_sgd.py --data_size=95 --test_size=45 --batch_size=5 --epochs=9 --learning_rate=0.5`
```
After epoch 1: train loss 0.2429 acc 96.0%, test loss 0.3187 acc 93.3%
After epoch 2: train loss 0.1853 acc 96.0%, test loss 0.2724 acc 93.3%
After epoch 3: train loss 0.1590 acc 96.0%, test loss 0.2525 acc 93.3%
After epoch 4: train loss 0.1428 acc 96.0%, test loss 0.2411 acc 93.3%
After epoch 5: train loss 0.1313 acc 98.0%, test loss 0.2335 acc 93.3%
After epoch 6: train loss 0.1225 acc 96.0%, test loss 0.2258 acc 93.3%
After epoch 7: train loss 0.1159 acc 96.0%, test loss 0.2220 acc 93.3%
After epoch 8: train loss 0.1105 acc 96.0%, test loss 0.2187 acc 93.3%
After epoch 9: train loss 0.1061 acc 96.0%, test loss 0.2163 acc 93.3%
Learned weights -0.61 3.61 0.12
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/logistic_regression_sgd_2.svgz)
- `python3 logistic_regression_sgd.py --data_size=95 --test_size=45 --batch_size=1 --epochs=9 --learning_rate=0.7`
```
After epoch 1: train loss 0.1141 acc 96.0%, test loss 0.2268 acc 93.3%
After epoch 2: train loss 0.0867 acc 96.0%, test loss 0.2150 acc 91.1%
After epoch 3: train loss 0.0797 acc 98.0%, test loss 0.2320 acc 88.9%
After epoch 4: train loss 0.0753 acc 96.0%, test loss 0.2224 acc 88.9%
After epoch 5: train loss 0.0692 acc 96.0%, test loss 0.2154 acc 88.9%
After epoch 6: train loss 0.0749 acc 98.0%, test loss 0.2458 acc 88.9%
After epoch 7: train loss 0.0638 acc 96.0%, test loss 0.2190 acc 88.9%
After epoch 8: train loss 0.0644 acc 98.0%, test loss 0.2341 acc 88.9%
After epoch 9: train loss 0.0663 acc 98.0%, test loss 0.2490 acc 88.9%
Learned weights -1.07 7.33 -0.40
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/logistic_regression_sgd_3.svgz)
#### Tests End:
```

### Assignment: perceptron
#### Date: Deadline: Oct 31, 7:59 a.m.
#### Points: 2 points
#### Tests: perceptron_tests

Starting with the [perceptron.py](https://github.com/ufal/npfl129/tree/master/labs/03/perceptron.py)
template, implement the perceptron algorithm.

#### Tests Start: perceptron_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 perceptron.py --data_size=100 --seed=17`
```
Learned weights 4.10 2.94 -1.00
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/perceptron_1.svgz)
- `python3 perceptron.py --data_size=50 --seed=320`
```
Learned weights -2.30 -1.96 -2.00
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/perceptron_2.svgz)
- `python3 perceptron.py --data_size=200 --seed=92`
```
Learned weights 4.43 1.54 -2.00
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/perceptron_3.svgz)
#### Tests End:
```

### Assignment: mlp_classification_sgd
#### Date: Deadline: Nov 7, 7:59 a.m.
#### Points: 6 points
#### Examples: mlp_classification_sgd_examples
#### Tests: mlp_classification_sgd_tests

Starting with the [mlp_classification_sgd.py](https://github.com/ufal/npfl129/tree/master/labs/04/mlp_classification_sgd.py),
implement minibatch SGD for multilayer perceptron classification.

#### Examples Start: mlp_classification_sgd_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 mlp_classification_sgd.py --epochs=10 --batch_size=10 --hidden_layer=20`
```
After epoch 1: train acc 79.7%, test acc 80.2%
After epoch 2: train acc 91.9%, test acc 88.3%
After epoch 3: train acc 92.4%, test acc 90.0%
After epoch 4: train acc 96.1%, test acc 93.1%
After epoch 5: train acc 95.3%, test acc 93.1%
After epoch 6: train acc 96.6%, test acc 93.9%
After epoch 7: train acc 97.3%, test acc 94.2%
After epoch 8: train acc 98.2%, test acc 94.9%
After epoch 9: train acc 98.1%, test acc 95.7%
After epoch 10: train acc 97.4%, test acc 95.1%
Learned parameters:
  -0.03 0.09 0.05 0.02 -0.07 -0.07 -0.09 0.07 0.02 0.04 -0.10 0.09 ...
  -0.07 0.12 0.33 -0.21 -0.16 -0.13 0.02 -0.14 0.01 -0.12 -0.02 -0.04 ...
  -0.00 -0.00 0.00 -0.00 0.00 0.00 -0.00 0.00 0.00 -0.00 0.00 0.00 ...
  0.02 -0.01 0.01 -0.03 0.02 -0.01 0.00 0.01 0.01 -0.01 ...
```
- `python3 mlp_classification_sgd.py --epochs=10 --batch_size=10 --hidden_layer=50`
```
After epoch 1: train acc 91.1%, test acc 89.2%
After epoch 2: train acc 95.9%, test acc 93.5%
After epoch 3: train acc 96.5%, test acc 95.2%
After epoch 4: train acc 96.1%, test acc 94.5%
After epoch 5: train acc 96.3%, test acc 93.5%
After epoch 6: train acc 98.3%, test acc 96.2%
After epoch 7: train acc 98.4%, test acc 96.4%
After epoch 8: train acc 98.3%, test acc 95.7%
After epoch 9: train acc 99.1%, test acc 97.4%
After epoch 10: train acc 98.8%, test acc 97.4%
Learned parameters:
  -0.03 0.09 0.05 0.02 -0.07 -0.07 -0.09 0.07 0.02 0.04 -0.10 0.09 ...
  0.01 0.10 -0.16 0.02 0.13 0.04 0.14 -0.01 0.05 -0.07 -0.08 0.02 ...
  0.00 0.00 -0.00 0.00 0.00 -0.00 -0.00 0.00 0.00 -0.00 -0.00 -0.00 ...
  0.01 -0.00 -0.00 0.00 0.00 0.00 -0.01 0.00 -0.00 -0.00 ...
```
- `python3 mlp_classification_sgd.py --epochs=10 --batch_size=10 --hidden_layer=200`
```
After epoch 1: train acc 95.4%, test acc 93.0%
After epoch 2: train acc 97.9%, test acc 96.6%
After epoch 3: train acc 98.8%, test acc 96.9%
After epoch 4: train acc 98.0%, test acc 95.4%
After epoch 5: train acc 99.6%, test acc 97.7%
After epoch 6: train acc 99.7%, test acc 98.0%
After epoch 7: train acc 97.4%, test acc 95.4%
After epoch 8: train acc 99.7%, test acc 97.5%
After epoch 9: train acc 99.8%, test acc 97.9%
After epoch 10: train acc 99.9%, test acc 97.9%
Learned parameters:
  -0.03 0.09 0.05 0.02 -0.07 -0.07 -0.09 0.07 0.02 0.04 -0.10 0.09 ...
  0.01 -0.09 0.04 -0.09 0.06 0.06 -0.05 -0.04 -0.00 0.02 -0.04 0.02 ...
  0.00 0.00 0.00 -0.00 0.00 -0.00 -0.00 -0.00 0.00 0.00 0.00 -0.00 ...
  -0.00 -0.01 -0.00 -0.00 0.00 -0.00 -0.00 0.00 0.01 0.00 ...
```
#### Examples End:
#### Tests Start: mlp_classification_sgd_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 mlp_classification_sgd.py --epochs=3 --batch_size=10 --hidden_layer=20`
```
After epoch 1: train acc 79.7%, test acc 80.2%
After epoch 2: train acc 91.9%, test acc 88.3%
After epoch 3: train acc 92.4%, test acc 90.0%
Learned parameters:
  -0.03 0.09 0.05 0.02 -0.07 -0.07 -0.09 0.07 0.02 0.04 -0.10 0.09 ...
  -0.09 0.07 0.21 -0.16 -0.15 -0.07 0.01 -0.09 0.05 -0.11 -0.02 -0.04 ...
  -0.00 -0.00 0.00 0.00 0.00 0.00 -0.00 0.00 0.00 -0.00 0.00 -0.00 ...
  0.01 -0.01 0.01 -0.02 0.01 -0.01 0.00 0.01 0.01 -0.01 ...
```
- `python3 mlp_classification_sgd.py --epochs=3 --batch_size=10 --hidden_layer=50`
```
After epoch 1: train acc 91.1%, test acc 89.2%
After epoch 2: train acc 95.9%, test acc 93.5%
After epoch 3: train acc 96.5%, test acc 95.2%
Learned parameters:
  -0.03 0.09 0.05 0.02 -0.07 -0.07 -0.09 0.07 0.02 0.04 -0.10 0.09 ...
  0.01 0.06 -0.13 0.04 0.11 0.04 0.13 0.01 0.05 -0.05 -0.07 0.02 ...
  0.00 0.00 -0.00 0.00 -0.00 0.00 -0.00 0.00 0.00 -0.00 -0.00 -0.00 ...
  0.01 0.00 -0.00 0.00 -0.00 0.00 -0.01 -0.00 -0.00 0.00 ...
```
- `python3 mlp_classification_sgd.py --epochs=3 --batch_size=10 --hidden_layer=200`
```
After epoch 1: train acc 95.4%, test acc 93.0%
After epoch 2: train acc 97.9%, test acc 96.6%
After epoch 3: train acc 98.8%, test acc 96.9%
Learned parameters:
  -0.03 0.09 0.05 0.02 -0.07 -0.07 -0.09 0.07 0.02 0.04 -0.10 0.09 ...
  0.01 -0.09 0.04 -0.09 0.06 0.06 -0.05 -0.04 -0.00 0.02 -0.04 0.02 ...
  0.00 0.00 -0.00 -0.00 0.00 -0.00 0.00 -0.00 0.00 0.00 -0.00 -0.00 ...
  -0.00 -0.00 -0.00 -0.00 0.00 -0.00 -0.00 0.00 0.00 0.00 ...
```
- `python3 mlp_classification_sgd.py --epochs=1 --batch_size=1  --hidden_layer=200 --test_size=1597`
```
After epoch 1: train acc 74.0%, test acc 68.7%
Learned parameters:
  -0.03 0.09 0.05 0.02 -0.07 -0.07 -0.09 0.07 0.02 0.04 -0.10 0.09 ...
  0.01 -0.09 0.04 -0.09 0.06 0.06 -0.05 -0.04 -0.00 0.02 -0.04 0.02 ...
  0.00 0.00 -0.00 -0.00 -0.00 -0.00 -0.00 -0.00 0.00 -0.00 -0.00 -0.00 ...
  -0.02 0.01 -0.00 -0.02 0.02 -0.00 0.00 -0.02 0.02 0.01 ...
```
#### Tests End:


```
### Assignment: softmax_classification_sgd
#### Date: Deadline: Nov 7, 7:59 a.m.
#### Points: 3 points
#### Examples: softmax_classification_sgd_examples
#### Tests: softmax_classification_sgd_tests

Starting with the [softmax_classification_sgd.py](https://github.com/ufal/npfl129/tree/master/labs/04/softmax_classification_sgd.py),
implement minibatch SGD for multinomial logistic regression.

#### Examples Start: softmax_classification_sgd_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 softmax_classification_sgd.py --batch_size=10 --epochs=10 --learning_rate=0.005`
```
After epoch 1: train loss 0.3130 acc 90.8%, test loss 0.3529 acc 88.7%
After epoch 2: train loss 0.2134 acc 93.9%, test loss 0.2450 acc 92.5%
After epoch 3: train loss 0.1366 acc 96.8%, test loss 0.1735 acc 94.6%
After epoch 4: train loss 0.1374 acc 96.2%, test loss 0.1705 acc 94.0%
After epoch 5: train loss 0.1169 acc 97.2%, test loss 0.1667 acc 95.1%
After epoch 6: train loss 0.0978 acc 97.5%, test loss 0.1340 acc 96.1%
After epoch 7: train loss 0.0878 acc 98.0%, test loss 0.1366 acc 95.9%
After epoch 8: train loss 0.0889 acc 97.5%, test loss 0.1515 acc 95.1%
After epoch 9: train loss 0.0819 acc 98.0%, test loss 0.1336 acc 96.5%
After epoch 10: train loss 0.0801 acc 97.9%, test loss 0.1342 acc 96.4%
Learned weights:
  -0.03 -0.10 0.01 0.08 -0.05 0.01 -0.06 0.05 0.07 -0.10 ...
  0.09 0.07 -0.15 -0.02 -0.21 0.13 -0.01 -0.06 0.02 -0.07 ...
  0.05 0.08 0.01 -0.03 -0.05 0.06 0.04 -0.10 -0.03 0.09 ...
  0.02 -0.03 -0.02 0.11 0.16 0.09 -0.06 0.06 -0.09 0.05 ...
  -0.07 -0.07 -0.10 -0.07 -0.10 -0.13 -0.09 0.03 -0.04 0.02 ...
  -0.07 -0.04 0.20 0.05 -0.02 0.12 0.06 0.04 -0.04 0.01 ...
  -0.09 -0.04 -0.14 -0.09 -0.02 -0.08 -0.09 0.05 0.05 -0.03 ...
  0.07 0.01 0.05 -0.01 0.06 -0.01 0.13 -0.04 0.03 -0.02 ...
  0.02 -0.02 0.01 -0.08 0.03 0.01 -0.10 -0.03 0.08 -0.05 ...
  0.04 -0.05 -0.07 0.09 -0.00 -0.05 0.10 -0.09 -0.01 0.01 ...
```
- `python3 softmax_classification_sgd.py --batch_size=1 --epochs=10 --learning_rate=0.005 --test_size=1597`
```
After epoch 1: train loss 1.7683 acc 73.5%, test loss 2.0028 acc 72.2%
After epoch 2: train loss 0.7731 acc 88.5%, test loss 1.5349 acc 77.8%
After epoch 3: train loss 1.2189 acc 82.5%, test loss 2.0718 acc 73.7%
After epoch 4: train loss 1.1752 acc 89.5%, test loss 2.3474 acc 79.0%
After epoch 5: train loss 0.2969 acc 95.5%, test loss 1.0299 acc 86.0%
After epoch 6: train loss 0.2176 acc 96.0%, test loss 0.9374 acc 86.7%
After epoch 7: train loss 0.1214 acc 97.5%, test loss 0.8018 acc 87.7%
After epoch 8: train loss 0.0178 acc 99.0%, test loss 0.5969 acc 90.4%
After epoch 9: train loss 0.2188 acc 94.0%, test loss 1.2211 acc 83.0%
After epoch 10: train loss 0.0054 acc 100.0%, test loss 0.6710 acc 89.8%
Learned weights:
  -0.03 -0.10 0.05 0.12 0.09 0.00 -0.07 0.05 0.07 -0.15 ...
  0.09 0.10 -0.31 -0.21 -0.55 0.21 -0.08 -0.06 0.02 -0.11 ...
  0.05 0.07 0.14 -0.01 -0.15 0.02 0.03 -0.10 -0.04 0.28 ...
  0.02 -0.02 0.11 0.28 0.13 0.11 -0.12 0.06 -0.08 0.19 ...
  -0.07 -0.09 -0.10 -0.32 -0.27 -0.50 -0.21 0.04 -0.04 0.06 ...
  -0.07 -0.07 0.42 0.18 0.11 0.51 0.13 0.04 -0.03 0.12 ...
  -0.09 -0.05 -0.31 -0.16 0.15 -0.02 -0.12 0.05 0.05 -0.10 ...
  0.07 0.02 0.05 0.09 0.16 0.05 0.20 -0.08 0.03 -0.10 ...
  0.02 -0.02 -0.12 -0.06 0.07 -0.09 -0.06 -0.03 0.08 -0.18 ...
  0.04 -0.04 -0.14 0.08 0.08 -0.15 0.24 -0.08 -0.01 -0.11 ...
```
- `python3 softmax_classification_sgd.py --batch_size=100 --epochs=10 --learning_rate=0.05`
```
After epoch 1: train loss 4.1126 acc 77.8%, test loss 4.2883 acc 75.5%
After epoch 2: train loss 0.4290 acc 90.5%, test loss 0.5414 acc 89.8%
After epoch 3: train loss 0.6189 acc 88.0%, test loss 0.5752 acc 89.2%
After epoch 4: train loss 0.3084 acc 91.9%, test loss 0.3482 acc 91.3%
After epoch 5: train loss 0.2757 acc 93.2%, test loss 0.3792 acc 91.3%
After epoch 6: train loss 0.2559 acc 92.7%, test loss 0.3718 acc 91.8%
After epoch 7: train loss 0.1164 acc 96.8%, test loss 0.1761 acc 95.1%
After epoch 8: train loss 0.2891 acc 91.5%, test loss 0.4110 acc 90.2%
After epoch 9: train loss 0.1256 acc 96.4%, test loss 0.1977 acc 94.9%
After epoch 10: train loss 0.1239 acc 96.3%, test loss 0.1847 acc 95.0%
Learned weights:
  -0.03 -0.10 -0.05 0.07 -0.08 -0.04 -0.06 0.05 0.07 -0.12 ...
  0.09 0.05 -0.24 -0.03 -0.25 0.16 -0.01 -0.06 0.02 -0.13 ...
  0.05 0.10 0.05 -0.02 -0.06 0.04 0.03 -0.10 -0.03 0.16 ...
  0.02 -0.03 0.03 0.15 0.25 0.13 -0.09 0.06 -0.09 0.11 ...
  -0.07 -0.08 -0.13 -0.10 -0.10 -0.18 -0.11 0.03 -0.04 0.00 ...
  -0.07 -0.02 0.32 0.06 0.03 0.23 0.10 0.04 -0.03 0.03 ...
  -0.09 -0.04 -0.18 -0.12 -0.01 -0.12 -0.10 0.05 0.04 -0.06 ...
  0.07 0.01 0.10 0.00 0.05 0.05 0.20 -0.02 0.03 -0.02 ...
  0.02 -0.03 -0.04 -0.12 0.02 -0.02 -0.15 -0.04 0.08 -0.08 ...
  0.04 -0.06 -0.06 0.12 -0.04 -0.10 0.12 -0.09 -0.01 0.01 ...
```
#### Examples End:
#### Tests Start: softmax_classification_sgd_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 softmax_classification_sgd.py --batch_size=10  --epochs=2 --learning_rate=0.005`
```
After epoch 1: train loss 0.3130 acc 90.8%, test loss 0.3529 acc 88.7%
After epoch 2: train loss 0.2134 acc 93.9%, test loss 0.2450 acc 92.5%
Learned weights:
  -0.03 -0.10 0.01 0.06 -0.07 0.04 -0.05 0.05 0.07 -0.10 ...
  0.09 0.08 -0.12 -0.08 -0.10 0.09 -0.03 -0.06 0.02 -0.01 ...
  0.05 0.07 0.01 -0.03 -0.05 0.06 0.04 -0.10 -0.03 0.08 ...
  0.02 -0.05 -0.01 0.10 0.11 0.09 -0.05 0.06 -0.09 0.04 ...
  -0.07 -0.07 -0.10 -0.01 -0.06 -0.07 -0.08 0.04 -0.04 0.01 ...
  -0.07 -0.05 0.14 0.06 0.02 0.14 0.05 0.04 -0.04 0.03 ...
  -0.09 -0.04 -0.11 -0.06 -0.04 -0.10 -0.09 0.05 0.05 -0.01 ...
  0.07 0.01 0.02 -0.04 0.04 -0.01 0.11 -0.06 0.03 -0.03 ...
  0.02 -0.02 0.01 -0.03 0.00 -0.03 -0.09 -0.03 0.08 -0.07 ...
  0.04 -0.04 -0.05 0.05 -0.04 -0.05 0.09 -0.08 -0.01 -0.04 ...
```
- `python3 softmax_classification_sgd.py --batch_size=1   --epochs=1 --learning_rate=0.005 --test_size=1597`
```
After epoch 1: train loss 1.7683 acc 73.5%, test loss 2.0028 acc 72.2%
Learned weights:
  -0.03 -0.10 0.03 0.08 0.03 0.03 -0.07 0.05 0.07 -0.15 ...
  0.09 0.08 -0.25 -0.15 -0.17 0.11 -0.00 -0.06 0.02 -0.05 ...
  0.05 0.06 0.07 0.04 -0.12 0.11 0.07 -0.10 -0.03 0.16 ...
  0.02 -0.03 0.03 0.14 0.03 0.08 -0.09 0.06 -0.09 0.09 ...
  -0.07 -0.08 -0.22 -0.07 -0.11 -0.27 -0.13 0.04 -0.04 -0.00 ...
  -0.07 -0.08 0.17 0.16 0.17 0.39 0.07 0.04 -0.03 0.03 ...
  -0.09 -0.04 -0.13 -0.10 -0.03 -0.16 -0.09 0.05 0.05 -0.03 ...
  0.07 0.02 0.10 0.03 0.09 -0.05 0.13 -0.08 0.03 -0.05 ...
  0.02 0.00 0.02 -0.17 -0.01 -0.04 -0.12 -0.03 0.08 -0.09 ...
  0.04 -0.04 -0.02 0.06 -0.07 -0.05 0.16 -0.08 -0.01 -0.01 ...
```
- `python3 softmax_classification_sgd.py --batch_size=100 --epochs=3 --learning_rate=0.05`
```
After epoch 1: train loss 4.1126 acc 77.8%, test loss 4.2883 acc 75.5%
After epoch 2: train loss 0.4290 acc 90.5%, test loss 0.5414 acc 89.8%
After epoch 3: train loss 0.6189 acc 88.0%, test loss 0.5752 acc 89.2%
Learned weights:
  -0.03 -0.10 -0.04 0.08 -0.07 -0.02 -0.05 0.05 0.07 -0.12 ...
  0.09 0.06 -0.23 -0.08 -0.12 0.11 -0.04 -0.06 0.02 -0.09 ...
  0.05 0.09 0.07 -0.01 -0.08 0.01 0.02 -0.10 -0.03 0.16 ...
  0.02 -0.04 0.03 0.15 0.18 0.09 -0.09 0.06 -0.09 0.10 ...
  -0.07 -0.07 -0.14 -0.07 -0.07 -0.13 -0.11 0.03 -0.04 -0.01 ...
  -0.07 -0.03 0.28 0.07 0.06 0.29 0.11 0.04 -0.03 0.08 ...
  -0.09 -0.04 -0.16 -0.09 -0.04 -0.13 -0.09 0.05 0.04 -0.05 ...
  0.07 0.01 0.06 -0.01 0.05 0.06 0.18 -0.04 0.03 -0.03 ...
  0.02 -0.03 -0.03 -0.08 -0.00 -0.03 -0.12 -0.04 0.08 -0.12 ...
  0.04 -0.05 -0.04 0.06 -0.09 -0.09 0.12 -0.09 -0.01 -0.03 ...
```
#### Tests End:
```

### Assignment: multilabel_classification_sgd
#### Date: Deadline: Nov 14, 7:59 a.m.
#### Points: 3 points
#### Examples: multilabel_classification_sgd_examples
#### Tests: multilabel_classification_sgd_tests

Starting with the [multilabel_classification_sgd.py](https://github.com/ufal/npfl129/tree/master/labs/05/multilabel_classification_sgd.py),
implement minibatch SGD for multi-label classification and
manually compute micro-averaged and macro-averaged $F_1$-score.

#### Examples Start: multilabel_classification_sgd_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=9 --classes=5`
```
After epoch 1: train F1 micro 56.45% macro 46.71%, test F1 micro 58.25% macro 43.9%
After epoch 2: train F1 micro 71.46% macro 59.47%, test F1 micro 73.77% macro 60.3%
After epoch 3: train F1 micro 73.06% macro 61.02%, test F1 micro 71.71% macro 56.8%
After epoch 4: train F1 micro 77.30% macro 66.48%, test F1 micro 76.19% macro 64.1%
After epoch 5: train F1 micro 76.05% macro 67.34%, test F1 micro 74.46% macro 61.4%
After epoch 6: train F1 micro 78.22% macro 73.24%, test F1 micro 77.40% macro 66.1%
After epoch 7: train F1 micro 78.13% macro 73.33%, test F1 micro 74.41% macro 61.7%
After epoch 8: train F1 micro 78.92% macro 74.73%, test F1 micro 76.78% macro 66.9%
After epoch 9: train F1 micro 80.76% macro 76.31%, test F1 micro 78.18% macro 68.3%
Learned weights:
  -0.09 -0.17 -0.16 -0.01 0.09 0.01 0.04 -0.09 0.04 0.07 ...
  -0.08 0.09 0.02 -0.07 -0.08 -0.13 -0.07 0.09 0.06 0.01 ...
  0.20 0.25 0.09 0.00 0.02 -0.18 -0.18 -0.15 0.06 0.07 ...
  0.06 -0.04 -0.07 -0.01 0.10 0.13 0.10 0.17 0.20 -0.01 ...
  0.06 -0.11 -0.12 -0.05 -0.20 0.04 -0.01 -0.03 -0.16 -0.11 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=9 --classes=10`
```
After epoch 1: train F1 micro 20.14% macro 9.95%, test F1 micro 21.57% macro 10.4%
After epoch 2: train F1 micro 11.29% macro 7.35%, test F1 micro 14.45% macro 8.8%
After epoch 3: train F1 micro 41.53% macro 26.29%, test F1 micro 33.54% macro 20.4%
After epoch 4: train F1 micro 44.23% macro 30.24%, test F1 micro 37.85% macro 24.4%
After epoch 5: train F1 micro 43.23% macro 29.85%, test F1 micro 42.37% macro 28.3%
After epoch 6: train F1 micro 49.53% macro 35.63%, test F1 micro 46.53% macro 32.2%
After epoch 7: train F1 micro 55.69% macro 40.36%, test F1 micro 48.21% macro 33.8%
After epoch 8: train F1 micro 52.47% macro 37.65%, test F1 micro 46.53% macro 31.9%
After epoch 9: train F1 micro 59.89% macro 43.27%, test F1 micro 53.44% macro 37.5%
Learned weights:
  -0.02 -0.04 -0.02 -0.08 -0.04 -0.10 0.12 0.04 -0.06 -0.15 ...
  0.18 0.04 -0.10 -0.06 0.15 -0.06 -0.08 0.05 0.05 0.05 ...
  0.13 -0.02 -0.20 -0.20 -0.01 0.13 -0.06 -0.15 0.09 -0.08 ...
  -0.05 -0.08 0.11 0.12 0.13 -0.07 0.05 -0.22 -0.02 -0.02 ...
  -0.09 -0.14 -0.00 -0.02 -0.10 -0.05 -0.09 -0.08 -0.06 0.07 ...
  -0.10 -0.01 0.11 0.03 0.03 0.04 0.05 -0.11 -0.04 -0.10 ...
  -0.16 -0.09 -0.13 -0.11 -0.10 -0.20 -0.04 -0.00 0.04 -0.08 ...
  -0.03 0.05 -0.21 -0.09 -0.12 0.03 -0.13 -0.09 -0.02 0.13 ...
  0.05 0.07 0.08 0.04 -0.18 -0.11 -0.09 0.18 -0.09 -0.07 ...
  0.04 -0.10 0.00 -0.07 -0.15 0.17 -0.03 -0.12 -0.12 -0.16 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=5 --epochs=9 --classes=5 --learning_rate=0.02`
```
After epoch 1: train F1 micro 60.66% macro 47.96%, test F1 micro 60.82% macro 46.6%
After epoch 2: train F1 micro 79.28% macro 77.99%, test F1 micro 77.65% macro 71.1%
After epoch 3: train F1 micro 80.27% macro 74.86%, test F1 micro 79.57% macro 69.6%
After epoch 4: train F1 micro 81.22% macro 79.85%, test F1 micro 77.41% macro 70.1%
After epoch 5: train F1 micro 80.50% macro 78.76%, test F1 micro 72.54% macro 65.1%
After epoch 6: train F1 micro 82.86% macro 81.46%, test F1 micro 75.62% macro 69.2%
After epoch 7: train F1 micro 81.19% macro 79.54%, test F1 micro 72.51% macro 65.3%
After epoch 8: train F1 micro 81.37% macro 79.59%, test F1 micro 75.06% macro 68.9%
After epoch 9: train F1 micro 83.83% macro 82.38%, test F1 micro 79.74% macro 74.3%
Learned weights:
  -0.18 -0.31 -0.23 0.05 0.12 -0.02 0.09 -0.25 0.21 0.16 ...
  -0.21 0.18 -0.12 -0.08 -0.13 -0.17 -0.12 0.15 0.10 0.04 ...
  0.47 0.32 0.13 0.01 0.09 -0.36 -0.29 -0.26 0.27 0.14 ...
  0.12 -0.07 -0.11 0.04 0.28 0.21 0.11 0.28 0.39 0.04 ...
  0.22 -0.24 -0.26 -0.03 -0.48 0.06 -0.10 0.01 -0.28 -0.14 ...
```
#### Examples End:
#### Tests Start: multilabel_classification_sgd_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=2 --classes=5`
```
After epoch 1: train F1 micro 56.45% macro 46.71%, test F1 micro 58.25% macro 43.9%
After epoch 2: train F1 micro 71.46% macro 59.47%, test F1 micro 73.77% macro 60.3%
Learned weights:
  -0.05 -0.11 -0.12 -0.05 0.04 0.04 0.02 0.01 -0.05 0.03 ...
  0.05 -0.01 0.09 -0.05 -0.06 -0.08 -0.05 0.02 0.03 0.00 ...
  0.10 0.16 0.08 0.01 -0.02 -0.05 -0.11 -0.09 -0.04 0.05 ...
  0.03 0.00 -0.06 -0.01 0.01 0.06 0.10 0.08 0.12 0.01 ...
  -0.03 -0.02 -0.08 -0.05 -0.07 -0.05 0.06 -0.03 -0.09 -0.09 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=2 --classes=10`
```
After epoch 1: train F1 micro 20.14% macro 9.95%, test F1 micro 21.57% macro 10.4%
After epoch 2: train F1 micro 11.29% macro 7.35%, test F1 micro 14.45% macro 8.8%
Learned weights:
  -0.04 -0.09 -0.01 -0.01 -0.09 0.02 0.01 0.04 0.02 -0.11 ...
  0.12 0.07 -0.09 -0.07 0.04 0.02 -0.06 -0.03 0.03 0.05 ...
  0.05 0.03 -0.11 -0.13 -0.09 0.08 0.02 -0.14 -0.01 -0.00 ...
  -0.03 -0.07 0.00 0.09 0.08 0.01 -0.01 -0.04 -0.08 -0.02 ...
  -0.11 -0.11 -0.04 0.04 -0.11 -0.03 -0.08 -0.03 -0.07 0.03 ...
  -0.11 -0.07 0.04 0.04 -0.00 0.04 0.00 -0.03 -0.06 -0.05 ...
  -0.14 -0.08 -0.12 -0.09 -0.11 -0.15 -0.09 -0.01 0.01 -0.05 ...
  0.04 0.00 -0.08 -0.10 -0.06 -0.04 -0.01 -0.10 -0.00 0.02 ...
  0.03 0.01 0.04 0.03 -0.06 -0.10 -0.09 0.04 0.02 -0.10 ...
  0.04 -0.06 -0.07 -0.03 -0.09 0.04 0.05 -0.09 -0.04 -0.10 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=5 --epochs=2 --classes=5 --learning_rate=0.02`
```
After epoch 1: train F1 micro 60.66% macro 47.96%, test F1 micro 60.82% macro 46.6%
After epoch 2: train F1 micro 79.28% macro 77.99%, test F1 micro 77.65% macro 71.1%
Learned weights:
  -0.08 -0.15 -0.14 -0.01 0.09 0.03 0.04 -0.08 0.03 0.08 ...
  -0.06 0.09 0.04 -0.06 -0.08 -0.13 -0.06 0.11 0.07 0.01 ...
  0.21 0.28 0.12 0.03 0.02 -0.16 -0.16 -0.14 0.06 0.13 ...
  0.07 -0.00 -0.04 0.00 0.12 0.13 0.11 0.19 0.21 0.03 ...
  0.07 -0.10 -0.10 -0.04 -0.19 0.05 0.01 -0.03 -0.15 -0.10 ...
```
#### Tests End:
```

### Assignment: kernel_linear_regression
#### Date: Deadline: Nov 21, 7:59 a.m.
#### Points: 5 points
#### Examples: kernel_linear_regression_examples
#### Tests: kernel_linear_regression_tests

Starting with the [kernel_linear_regression.py](https://github.com/ufal/npfl129/tree/master/labs/06/kernel_linear_regression.py),
implement kernel linear regression training using SGD
on the dual formulation. You should support _polynomial_
and _Gaussian_ kernels and also L2 regularization.

#### Examples Start: kernel_linear_regression_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 kernel_linear_regression.py --batch_size=5 --kernel=poly --kernel_degree=3 --learning_rate=0.1`
```
After epoch 10: train RMSE 0.69, test RMSE 0.61
After epoch 20: train RMSE 0.61, test RMSE 0.65
After epoch 30: train RMSE 0.56, test RMSE 0.72
After epoch 40: train RMSE 0.53, test RMSE 0.77
After epoch 50: train RMSE 0.51, test RMSE 0.84
After epoch 60: train RMSE 0.49, test RMSE 0.91
After epoch 70: train RMSE 0.48, test RMSE 0.98
After epoch 80: train RMSE 0.48, test RMSE 1.01
After epoch 90: train RMSE 0.47, test RMSE 1.04
After epoch 100: train RMSE 0.47, test RMSE 1.05
After epoch 110: train RMSE 0.48, test RMSE 1.09
After epoch 120: train RMSE 0.47, test RMSE 1.12
After epoch 130: train RMSE 0.47, test RMSE 1.12
After epoch 140: train RMSE 0.47, test RMSE 1.13
After epoch 150: train RMSE 0.47, test RMSE 1.13
After epoch 160: train RMSE 0.47, test RMSE 1.10
After epoch 170: train RMSE 0.47, test RMSE 1.13
After epoch 180: train RMSE 0.47, test RMSE 1.16
After epoch 190: train RMSE 0.47, test RMSE 1.14
After epoch 200: train RMSE 0.47, test RMSE 1.14
Learned betas -2.28 -1.44 0.55 2.41 1.17 1.48 3.39 2.52 0.96 1.62 0.11 -0.37 -0.20 -2.91 -3.15 ...
Learned bias 0.44076460113546156
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_1.svgz)
- `python3 kernel_linear_regression.py --batch_size=1 --kernel=poly --kernel_degree=5 --learning_rate=0.05`
```
After epoch 10: train RMSE 0.63, test RMSE 1.61
After epoch 20: train RMSE 0.54, test RMSE 0.90
After epoch 30: train RMSE 0.49, test RMSE 1.11
After epoch 40: train RMSE 0.46, test RMSE 1.01
After epoch 50: train RMSE 0.47, test RMSE 0.72
After epoch 60: train RMSE 0.44, test RMSE 0.89
After epoch 70: train RMSE 0.46, test RMSE 1.03
After epoch 80: train RMSE 0.41, test RMSE 0.86
After epoch 90: train RMSE 0.44, test RMSE 0.65
After epoch 100: train RMSE 0.51, test RMSE 0.39
After epoch 110: train RMSE 0.39, test RMSE 0.61
After epoch 120: train RMSE 0.43, test RMSE 0.78
After epoch 130: train RMSE 0.36, test RMSE 0.54
After epoch 140: train RMSE 0.36, test RMSE 0.52
After epoch 150: train RMSE 0.40, test RMSE 0.51
After epoch 160: train RMSE 0.36, test RMSE 0.51
After epoch 170: train RMSE 0.34, test RMSE 0.29
After epoch 180: train RMSE 0.31, test RMSE 0.28
After epoch 190: train RMSE 0.31, test RMSE 0.25
After epoch 200: train RMSE 0.38, test RMSE 0.34
Learned betas -5.90 -4.93 0.06 4.67 1.48 2.01 7.45 5.76 2.26 4.14 0.74 0.20 1.17 -5.26 -5.86 ...
Learned bias 0.48895858087675187
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_2.svgz)
- `python3 kernel_linear_regression.py --batch_size=1 --kernel=poly --kernel_degree=5 --learning_rate=0.05 --kernel_gamma=0.15`
```
After epoch 10: train RMSE 0.80, test RMSE 0.66
After epoch 20: train RMSE 0.77, test RMSE 0.65
After epoch 30: train RMSE 0.76, test RMSE 0.63
After epoch 40: train RMSE 0.77, test RMSE 0.66
After epoch 50: train RMSE 0.75, test RMSE 0.63
After epoch 60: train RMSE 0.74, test RMSE 0.62
After epoch 70: train RMSE 0.72, test RMSE 0.61
After epoch 80: train RMSE 0.71, test RMSE 0.60
After epoch 90: train RMSE 0.72, test RMSE 0.63
After epoch 100: train RMSE 0.69, test RMSE 0.60
After epoch 110: train RMSE 0.73, test RMSE 0.67
After epoch 120: train RMSE 0.72, test RMSE 0.65
After epoch 130: train RMSE 0.70, test RMSE 0.62
After epoch 140: train RMSE 0.66, test RMSE 0.60
After epoch 150: train RMSE 0.67, test RMSE 0.62
After epoch 160: train RMSE 0.66, test RMSE 0.61
After epoch 170: train RMSE 0.64, test RMSE 0.62
After epoch 180: train RMSE 0.64, test RMSE 0.61
After epoch 190: train RMSE 0.63, test RMSE 0.62
After epoch 200: train RMSE 0.64, test RMSE 0.65
Learned betas 3.77 3.44 6.39 9.15 4.53 4.08 7.47 4.30 -0.32 0.70 -3.85 -5.11 -4.98 -11.87 -12.67 ...
Learned bias 0.3756022427815734
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_3.svgz)
- `python3 kernel_linear_regression.py --batch_size=1 --kernel=poly --kernel_degree=5 --learning_rate=0.05 --l2=0.02`
```
After epoch 10: train RMSE 0.63, test RMSE 1.52
After epoch 20: train RMSE 0.56, test RMSE 0.88
After epoch 30: train RMSE 0.51, test RMSE 1.11
After epoch 40: train RMSE 0.50, test RMSE 1.05
After epoch 50: train RMSE 0.50, test RMSE 0.85
After epoch 60: train RMSE 0.48, test RMSE 1.03
After epoch 70: train RMSE 0.52, test RMSE 1.28
After epoch 80: train RMSE 0.49, test RMSE 1.17
After epoch 90: train RMSE 0.50, test RMSE 0.95
After epoch 100: train RMSE 0.59, test RMSE 0.66
After epoch 110: train RMSE 0.53, test RMSE 1.07
After epoch 120: train RMSE 0.56, test RMSE 1.30
After epoch 130: train RMSE 0.50, test RMSE 1.13
After epoch 140: train RMSE 0.49, test RMSE 1.08
After epoch 150: train RMSE 0.57, test RMSE 1.10
After epoch 160: train RMSE 0.46, test RMSE 0.96
After epoch 170: train RMSE 0.50, test RMSE 0.96
After epoch 180: train RMSE 0.47, test RMSE 1.03
After epoch 190: train RMSE 0.47, test RMSE 1.04
After epoch 200: train RMSE 0.56, test RMSE 0.96
Learned betas -0.65 -0.47 -0.10 0.55 0.39 0.34 0.87 0.60 0.26 0.44 0.05 -0.00 0.05 -0.68 -0.76 ...
Learned bias 0.9258410067869733
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_4.svgz)
- `python3 kernel_linear_regression.py --batch_size=1 --kernel=rbf`
```
After epoch 10: train RMSE 0.78, test RMSE 0.66
After epoch 20: train RMSE 0.74, test RMSE 0.61
After epoch 30: train RMSE 0.71, test RMSE 0.58
After epoch 40: train RMSE 0.67, test RMSE 0.54
After epoch 50: train RMSE 0.64, test RMSE 0.52
After epoch 60: train RMSE 0.62, test RMSE 0.50
After epoch 70: train RMSE 0.59, test RMSE 0.48
After epoch 80: train RMSE 0.57, test RMSE 0.47
After epoch 90: train RMSE 0.55, test RMSE 0.46
After epoch 100: train RMSE 0.53, test RMSE 0.45
After epoch 110: train RMSE 0.51, test RMSE 0.45
After epoch 120: train RMSE 0.49, test RMSE 0.45
After epoch 130: train RMSE 0.48, test RMSE 0.45
After epoch 140: train RMSE 0.46, test RMSE 0.46
After epoch 150: train RMSE 0.45, test RMSE 0.46
After epoch 160: train RMSE 0.44, test RMSE 0.47
After epoch 170: train RMSE 0.43, test RMSE 0.48
After epoch 180: train RMSE 0.42, test RMSE 0.49
After epoch 190: train RMSE 0.41, test RMSE 0.50
After epoch 200: train RMSE 0.40, test RMSE 0.50
Learned betas 0.65 0.59 1.17 1.72 0.86 0.82 1.61 1.04 0.21 0.47 -0.31 -0.56 -0.46 -1.77 -1.88 ...
Learned bias 0.6512539914766637
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_5.svgz)
- `python3 kernel_linear_regression.py --batch_size=1 --kernel=rbf --kernel_gamma=0.5`
```
After epoch 10: train RMSE 0.81, test RMSE 0.69
After epoch 20: train RMSE 0.80, test RMSE 0.67
After epoch 30: train RMSE 0.79, test RMSE 0.66
After epoch 40: train RMSE 0.78, test RMSE 0.65
After epoch 50: train RMSE 0.77, test RMSE 0.64
After epoch 60: train RMSE 0.77, test RMSE 0.64
After epoch 70: train RMSE 0.76, test RMSE 0.63
After epoch 80: train RMSE 0.75, test RMSE 0.62
After epoch 90: train RMSE 0.74, test RMSE 0.61
After epoch 100: train RMSE 0.74, test RMSE 0.61
After epoch 110: train RMSE 0.73, test RMSE 0.60
After epoch 120: train RMSE 0.72, test RMSE 0.60
After epoch 130: train RMSE 0.72, test RMSE 0.59
After epoch 140: train RMSE 0.71, test RMSE 0.58
After epoch 150: train RMSE 0.70, test RMSE 0.58
After epoch 160: train RMSE 0.70, test RMSE 0.57
After epoch 170: train RMSE 0.69, test RMSE 0.57
After epoch 180: train RMSE 0.69, test RMSE 0.56
After epoch 190: train RMSE 0.68, test RMSE 0.56
After epoch 200: train RMSE 0.68, test RMSE 0.56
Learned betas 1.45 1.28 1.74 2.17 1.18 1.01 1.67 0.98 0.03 0.19 -0.69 -1.02 -0.99 -2.36 -2.50 ...
Learned bias 0.6326715226190537
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_6.svgz)
- `python3 kernel_linear_regression.py --batch_size=2 --kernel=rbf --kernel_gamma=5`
```
After epoch 10: train RMSE 0.65, test RMSE 0.55
After epoch 20: train RMSE 0.52, test RMSE 0.40
After epoch 30: train RMSE 0.43, test RMSE 0.30
After epoch 40: train RMSE 0.36, test RMSE 0.22
After epoch 50: train RMSE 0.31, test RMSE 0.17
After epoch 60: train RMSE 0.27, test RMSE 0.15
After epoch 70: train RMSE 0.25, test RMSE 0.13
After epoch 80: train RMSE 0.24, test RMSE 0.13
After epoch 90: train RMSE 0.23, test RMSE 0.14
After epoch 100: train RMSE 0.22, test RMSE 0.15
After epoch 110: train RMSE 0.22, test RMSE 0.15
After epoch 120: train RMSE 0.22, test RMSE 0.16
After epoch 130: train RMSE 0.21, test RMSE 0.16
After epoch 140: train RMSE 0.21, test RMSE 0.17
After epoch 150: train RMSE 0.21, test RMSE 0.17
After epoch 160: train RMSE 0.21, test RMSE 0.17
After epoch 170: train RMSE 0.21, test RMSE 0.17
After epoch 180: train RMSE 0.21, test RMSE 0.17
After epoch 190: train RMSE 0.21, test RMSE 0.18
After epoch 200: train RMSE 0.21, test RMSE 0.18
Learned betas 0.21 0.08 0.29 0.51 0.06 0.05 0.49 0.27 -0.06 0.18 -0.09 -0.10 0.06 -0.49 -0.45 ...
Learned bias 0.7290386122306763
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_7.svgz)
- `python3 kernel_linear_regression.py --batch_size=1 --kernel=rbf --kernel_gamma=50`
```
After epoch 10: train RMSE 0.52, test RMSE 0.44
After epoch 20: train RMSE 0.36, test RMSE 0.29
After epoch 30: train RMSE 0.27, test RMSE 0.21
After epoch 40: train RMSE 0.23, test RMSE 0.18
After epoch 50: train RMSE 0.21, test RMSE 0.17
After epoch 60: train RMSE 0.20, test RMSE 0.17
After epoch 70: train RMSE 0.20, test RMSE 0.16
After epoch 80: train RMSE 0.20, test RMSE 0.16
After epoch 90: train RMSE 0.20, test RMSE 0.16
After epoch 100: train RMSE 0.19, test RMSE 0.16
After epoch 110: train RMSE 0.19, test RMSE 0.16
After epoch 120: train RMSE 0.19, test RMSE 0.16
After epoch 130: train RMSE 0.19, test RMSE 0.16
After epoch 140: train RMSE 0.19, test RMSE 0.16
After epoch 150: train RMSE 0.19, test RMSE 0.16
After epoch 160: train RMSE 0.19, test RMSE 0.16
After epoch 170: train RMSE 0.19, test RMSE 0.16
After epoch 180: train RMSE 0.19, test RMSE 0.16
After epoch 190: train RMSE 0.19, test RMSE 0.16
After epoch 200: train RMSE 0.19, test RMSE 0.16
Learned betas 0.61 0.03 0.28 0.67 -0.21 -0.21 0.69 0.28 -0.32 0.25 -0.17 -0.06 0.41 -0.59 -0.48 ...
Learned bias 0.8351544798239042
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_8.svgz)
- `python3 kernel_linear_regression.py --batch_size=1 --kernel=rbf --kernel_gamma=50 --l2=0.02`
```
After epoch 10: train RMSE 0.54, test RMSE 0.45
After epoch 20: train RMSE 0.39, test RMSE 0.32
After epoch 30: train RMSE 0.32, test RMSE 0.25
After epoch 40: train RMSE 0.28, test RMSE 0.22
After epoch 50: train RMSE 0.26, test RMSE 0.20
After epoch 60: train RMSE 0.25, test RMSE 0.19
After epoch 70: train RMSE 0.25, test RMSE 0.18
After epoch 80: train RMSE 0.24, test RMSE 0.18
After epoch 90: train RMSE 0.24, test RMSE 0.18
After epoch 100: train RMSE 0.24, test RMSE 0.18
After epoch 110: train RMSE 0.24, test RMSE 0.18
After epoch 120: train RMSE 0.24, test RMSE 0.18
After epoch 130: train RMSE 0.24, test RMSE 0.17
After epoch 140: train RMSE 0.24, test RMSE 0.17
After epoch 150: train RMSE 0.24, test RMSE 0.17
After epoch 160: train RMSE 0.24, test RMSE 0.17
After epoch 170: train RMSE 0.24, test RMSE 0.17
After epoch 180: train RMSE 0.24, test RMSE 0.17
After epoch 190: train RMSE 0.24, test RMSE 0.17
After epoch 200: train RMSE 0.24, test RMSE 0.17
Learned betas 0.35 0.11 0.22 0.38 -0.02 -0.03 0.35 0.16 -0.11 0.12 -0.09 -0.06 0.12 -0.34 -0.30 ...
Learned bias 0.9187854392321663
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/kernel_linear_regression_9.svgz)
#### Examples End:
#### Tests Start: kernel_linear_regression_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=5 --kernel=poly --kernel_degree=3 --learning_rate=0.1`
```
After epoch 10: train RMSE 0.69, test RMSE 0.61
After epoch 20: train RMSE 0.61, test RMSE 0.65
Learned betas 0.11 0.11 0.25 0.37 0.17 0.16 0.29 0.17 -0.02 0.02 -0.14 -0.20 -0.18 -0.45 -0.49 ...
Learned bias 0.4388019915399849
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=1 --kernel=poly --kernel_degree=5 --learning_rate=0.05`
```
After epoch 10: train RMSE 0.63, test RMSE 1.61
After epoch 20: train RMSE 0.54, test RMSE 0.90
Learned betas -0.81 -0.58 0.20 0.62 0.23 0.44 0.78 0.71 0.27 0.40 -0.00 0.06 -0.03 -0.63 -0.73 ...
Learned bias 0.44409714525206545
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=1 --kernel=poly --kernel_degree=5 --learning_rate=0.05 --kernel_gamma=0.15`
```
After epoch 10: train RMSE 0.80, test RMSE 0.66
After epoch 20: train RMSE 0.77, test RMSE 0.65
Learned betas 0.71 0.61 0.87 1.13 0.57 0.45 0.71 0.40 -0.09 -0.04 -0.50 -0.62 -0.64 -1.23 -1.38 ...
Learned bias 0.39859934926020046
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=1 --kernel=poly --kernel_degree=5 --learning_rate=0.05 --l2=0.02`
```
After epoch 10: train RMSE 0.63, test RMSE 1.52
After epoch 20: train RMSE 0.56, test RMSE 0.88
Learned betas -0.48 -0.32 0.13 0.38 0.13 0.28 0.49 0.44 0.16 0.24 -0.02 0.03 -0.04 -0.42 -0.47 ...
Learned bias 0.6096489059733282
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=1 --kernel=rbf`
```
After epoch 10: train RMSE 0.78, test RMSE 0.66
After epoch 20: train RMSE 0.74, test RMSE 0.61
Learned betas 0.21 0.19 0.24 0.27 0.17 0.14 0.20 0.13 0.03 0.04 -0.05 -0.08 -0.08 -0.22 -0.24 ...
Learned bias 0.6111050342939267
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=1 --kernel=rbf --kernel_gamma=0.5`
```
After epoch 10: train RMSE 0.81, test RMSE 0.69
After epoch 20: train RMSE 0.80, test RMSE 0.67
Learned betas 0.22 0.20 0.24 0.27 0.17 0.14 0.20 0.13 0.02 0.03 -0.06 -0.10 -0.09 -0.23 -0.25 ...
Learned bias 0.5619981553157737
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=2 --kernel=rbf --kernel_gamma=5`
```
After epoch 10: train RMSE 0.65, test RMSE 0.55
After epoch 20: train RMSE 0.52, test RMSE 0.40
Learned betas 0.11 0.10 0.12 0.13 0.08 0.07 0.11 0.07 0.03 0.03 -0.01 -0.02 -0.02 -0.08 -0.09 ...
Learned bias 0.7126228629963139
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=1 --kernel=rbf --kernel_gamma=50`
```
After epoch 10: train RMSE 0.52, test RMSE 0.44
After epoch 20: train RMSE 0.36, test RMSE 0.29
Learned betas 0.19 0.15 0.18 0.21 0.11 0.09 0.16 0.10 0.02 0.05 -0.02 -0.04 -0.02 -0.14 -0.15 ...
Learned bias 0.843378438496468
```
- `python3 kernel_linear_regression.py --epochs=20 --batch_size=1 --kernel=rbf --kernel_gamma=50 --l2=0.02`
```
After epoch 10: train RMSE 0.54, test RMSE 0.45
After epoch 20: train RMSE 0.39, test RMSE 0.32
Learned betas 0.17 0.14 0.16 0.19 0.10 0.08 0.15 0.10 0.02 0.05 -0.02 -0.04 -0.02 -0.13 -0.14 ...
Learned bias 0.8566622017610405
```
#### Tests End:
```
### Assignment: k_nearest_neighbors
#### Date: Deadline: ~~Nov 14~~ Nov 21, 7:59 a.m.
#### Points: 3 points
#### Tests: k_nearest_neighbors_tests

Starting with the [k_nearest_neighbors.py](https://github.com/ufal/npfl129/tree/master/labs/05/k_nearest_neighbors.py),
implement k-nearest neighbors algorithm for classifying MNIST, without using the
`sklearn.neighbors` module or `scipy.spatial` module in any way.

#### Tests Start: k_nearest_neighbors_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 k_nearest_neighbors.py --k=1 --p=2 --weights=uniform --test_size=500 --train_size=100`
```
K-nn accuracy for 1 nearest neighbors, L_2 metric, uniform weights: 73.60%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_1.svgz)
- `python3 k_nearest_neighbors.py --k=3 --p=2 --weights=uniform --test_size=500 --train_size=100`
```
K-nn accuracy for 3 nearest neighbors, L_2 metric, uniform weights: 66.80%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_2.svgz)
- `python3 k_nearest_neighbors.py --k=1 --p=2 --weights=uniform --test_size=500 --train_size=1000`
```
K-nn accuracy for 1 nearest neighbors, L_2 metric, uniform weights: 90.40%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_3.svgz)
- `python3 k_nearest_neighbors.py --k=5 --p=2 --weights=uniform --test_size=500 --train_size=1000`
```
K-nn accuracy for 5 nearest neighbors, L_2 metric, uniform weights: 88.40%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_4.svgz)
- `python3 k_nearest_neighbors.py --k=5 --p=1 --weights=uniform --test_size=500 --train_size=1000`
```
K-nn accuracy for 5 nearest neighbors, L_1 metric, uniform weights: 87.00%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_5.svgz)
- `python3 k_nearest_neighbors.py --k=5 --p=3 --weights=uniform --test_size=500 --train_size=1000`
```
K-nn accuracy for 5 nearest neighbors, L_3 metric, uniform weights: 89.40%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_6.svgz)
- `python3 k_nearest_neighbors.py --k=1 --p=2 --weights=uniform --test_size=500 --train_size=5000`
```
K-nn accuracy for 1 nearest neighbors, L_2 metric, uniform weights: 94.40%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_7.svgz)
- `python3 k_nearest_neighbors.py --k=9 --p=2 --weights=uniform --test_size=500 --train_size=5000`
```
K-nn accuracy for 9 nearest neighbors, L_2 metric, uniform weights: 92.80%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_8.svgz)
- `python3 k_nearest_neighbors.py --k=9 --p=2 --weights=inverse --test_size=500 --train_size=5000`
```
K-nn accuracy for 9 nearest neighbors, L_2 metric, inverse weights: 93.00%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_9.svgz)
- `python3 k_nearest_neighbors.py --k=9 --p=2 --weights=softmax --test_size=500 --train_size=5000`
```
K-nn accuracy for 9 nearest neighbors, L_2 metric, softmax weights: 94.00%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/k_nearest_neighbors_10.svgz)
#### Tests End:
```

### Assignment: smo_algorithm
#### Date: Deadline: Nov 28, 7:59 a.m.
#### Points: 7 points
#### Examples: smo_algorithm_examples
#### Tests: smo_algorithm_tests

Using the [smo_algorithm.py](https://github.com/ufal/npfl129/tree/master/labs/07/smo_algorithm.py)
template, implement the SMO algorithm for binary classification
using dual formulation of soft-margin SVM. The template contains
more detailed instructions.

#### Examples Start: smo_algorithm_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=1`
```
Iteration 100, train acc 88.0%, test acc 83.0%
Done, iteration 140, support vectors 41, train acc 88.0%, test acc 83.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_1.svgz)
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=3`
```
Iteration 100, train acc 89.0%, test acc 88.0%
Iteration 200, train acc 91.0%, test acc 86.0%
Iteration 300, train acc 86.0%, test acc 77.0%
Iteration 400, train acc 91.0%, test acc 84.0%
Iteration 500, train acc 88.0%, test acc 86.0%
Iteration 600, train acc 91.0%, test acc 86.0%
Iteration 700, train acc 91.0%, test acc 86.0%
Iteration 800, train acc 90.0%, test acc 86.0%
Iteration 900, train acc 91.0%, test acc 86.0%
Done, iteration 1000, support vectors 39, train acc 91.0%, test acc 86.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_2.svgz)
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=3 --C=5 --max_iterations=1500`
```
Iteration 100, train acc 85.0%, test acc 82.0%
Iteration 200, train acc 83.0%, test acc 83.0%
Iteration 300, train acc 84.0%, test acc 85.0%
Iteration 400, train acc 63.0%, test acc 66.0%
Iteration 500, train acc 89.0%, test acc 89.0%
Iteration 600, train acc 91.0%, test acc 89.0%
Iteration 700, train acc 89.0%, test acc 90.0%
Iteration 800, train acc 89.0%, test acc 89.0%
Iteration 900, train acc 55.0%, test acc 60.0%
Iteration 1000, train acc 91.0%, test acc 88.0%
Iteration 1100, train acc 91.0%, test acc 89.0%
Iteration 1200, train acc 90.0%, test acc 90.0%
Iteration 1300, train acc 91.0%, test acc 89.0%
Iteration 1400, train acc 89.0%, test acc 88.0%
Done, iteration 1500, support vectors 40, train acc 89.0%, test acc 90.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_3.svgz)
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=4 --kernel_gamma=0.6`
```
Iteration 100, train acc 65.0%, test acc 67.0%
Iteration 200, train acc 80.0%, test acc 80.0%
Iteration 300, train acc 92.0%, test acc 84.0%
Iteration 400, train acc 93.0%, test acc 85.0%
Iteration 500, train acc 92.0%, test acc 83.0%
Iteration 600, train acc 92.0%, test acc 86.0%
Iteration 700, train acc 92.0%, test acc 85.0%
Iteration 800, train acc 92.0%, test acc 85.0%
Iteration 900, train acc 92.0%, test acc 84.0%
Done, iteration 1000, support vectors 35, train acc 93.0%, test acc 86.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_4.svgz)
- `python3 smo_algorithm.py --kernel=rbf --kernel_gamma=1`
```
Iteration 100, train acc 92.0%, test acc 84.0%
Iteration 200, train acc 92.0%, test acc 84.0%
Iteration 300, train acc 92.0%, test acc 84.0%
Iteration 400, train acc 92.0%, test acc 84.0%
Done, iteration 483, support vectors 53, train acc 92.0%, test acc 84.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_5.svgz)
- `python3 smo_algorithm.py --kernel=rbf --kernel_gamma=0.1`
```
Done, iteration 87, support vectors 51, train acc 88.0%, test acc 85.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_6.svgz)
#### Examples End:
#### Tests Start: smo_algorithm_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=1`
```
Done, iteration 20, support vectors 48, train acc 84.0%, test acc 86.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=3`
```
Done, iteration 20, support vectors 70, train acc 85.0%, test acc 84.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=3 --C=5`
```
Done, iteration 20, support vectors 78, train acc 87.0%, test acc 86.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=4 --kernel_gamma=0.6`
```
Done, iteration 20, support vectors 55, train acc 84.0%, test acc 86.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=rbf --kernel_gamma=1`
```
Done, iteration 20, support vectors 67, train acc 92.0%, test acc 84.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=rbf --kernel_gamma=0.1`
```
Done, iteration 20, support vectors 53, train acc 85.0%, test acc 84.0%
```
#### Tests End:
```

### Assignment: smo_algorithm
#### Date: Deadline: Nov 28, 7:59 a.m.
#### Points: 7 points
#### Examples: smo_algorithm_examples
#### Tests: smo_algorithm_tests

Using the [smo_algorithm.py](https://github.com/ufal/npfl129/tree/master/labs/07/smo_algorithm.py)
template, implement the SMO algorithm for binary classification
using dual formulation of soft-margin SVM. The template contains
more detailed instructions.

#### Examples Start: smo_algorithm_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=1`
```
Iteration 100, train acc 88.0%, test acc 83.0%
Done, iteration 140, support vectors 41, train acc 88.0%, test acc 83.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_1.svgz)
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=3`
```
Iteration 100, train acc 89.0%, test acc 88.0%
Iteration 200, train acc 91.0%, test acc 86.0%
Iteration 300, train acc 86.0%, test acc 77.0%
Iteration 400, train acc 91.0%, test acc 84.0%
Iteration 500, train acc 88.0%, test acc 86.0%
Iteration 600, train acc 91.0%, test acc 86.0%
Iteration 700, train acc 91.0%, test acc 86.0%
Iteration 800, train acc 90.0%, test acc 86.0%
Iteration 900, train acc 91.0%, test acc 86.0%
Done, iteration 1000, support vectors 39, train acc 91.0%, test acc 86.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_2.svgz)
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=3 --C=5 --max_iterations=1500`
```
Iteration 100, train acc 85.0%, test acc 82.0%
Iteration 200, train acc 83.0%, test acc 83.0%
Iteration 300, train acc 84.0%, test acc 85.0%
Iteration 400, train acc 63.0%, test acc 66.0%
Iteration 500, train acc 89.0%, test acc 89.0%
Iteration 600, train acc 91.0%, test acc 89.0%
Iteration 700, train acc 89.0%, test acc 90.0%
Iteration 800, train acc 89.0%, test acc 89.0%
Iteration 900, train acc 55.0%, test acc 60.0%
Iteration 1000, train acc 91.0%, test acc 88.0%
Iteration 1100, train acc 91.0%, test acc 89.0%
Iteration 1200, train acc 90.0%, test acc 90.0%
Iteration 1300, train acc 91.0%, test acc 89.0%
Iteration 1400, train acc 89.0%, test acc 88.0%
Done, iteration 1500, support vectors 40, train acc 89.0%, test acc 90.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_3.svgz)
- `python3 smo_algorithm.py --kernel=poly --kernel_degree=4 --kernel_gamma=0.6`
```
Iteration 100, train acc 65.0%, test acc 67.0%
Iteration 200, train acc 80.0%, test acc 80.0%
Iteration 300, train acc 92.0%, test acc 84.0%
Iteration 400, train acc 93.0%, test acc 85.0%
Iteration 500, train acc 92.0%, test acc 83.0%
Iteration 600, train acc 92.0%, test acc 86.0%
Iteration 700, train acc 92.0%, test acc 85.0%
Iteration 800, train acc 92.0%, test acc 85.0%
Iteration 900, train acc 92.0%, test acc 84.0%
Done, iteration 1000, support vectors 35, train acc 93.0%, test acc 86.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_4.svgz)
- `python3 smo_algorithm.py --kernel=rbf --kernel_gamma=1`
```
Iteration 100, train acc 92.0%, test acc 84.0%
Iteration 200, train acc 92.0%, test acc 84.0%
Iteration 300, train acc 92.0%, test acc 84.0%
Iteration 400, train acc 92.0%, test acc 84.0%
Done, iteration 483, support vectors 53, train acc 92.0%, test acc 84.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_5.svgz)
- `python3 smo_algorithm.py --kernel=rbf --kernel_gamma=0.1`
```
Done, iteration 87, support vectors 51, train acc 88.0%, test acc 85.0%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/smo_algorithm_6.svgz)
#### Examples End:
#### Tests Start: smo_algorithm_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=1`
```
Done, iteration 20, support vectors 48, train acc 84.0%, test acc 86.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=3`
```
Done, iteration 20, support vectors 70, train acc 85.0%, test acc 84.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=3 --C=5`
```
Done, iteration 20, support vectors 78, train acc 87.0%, test acc 86.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=poly --kernel_degree=4 --kernel_gamma=0.6`
```
Done, iteration 20, support vectors 55, train acc 84.0%, test acc 86.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=rbf --kernel_gamma=1`
```
Done, iteration 20, support vectors 67, train acc 92.0%, test acc 84.0%
```
- `python3 smo_algorithm.py --max_iterations=20 --kernel=rbf --kernel_gamma=0.1`
```
Done, iteration 20, support vectors 53, train acc 85.0%, test acc 84.0%
```
#### Tests End:
```

### Assignment: metric_correlation
#### Date: Deadline: Dec 12, 7:59 a.m.
#### Points: 3 points
#### Tests: metric_correlation_tests

Using the [metric_correlation.py](https://github.com/ufal/npfl129/tree/master/labs/09/metric_correlation.py)
template, find a $\beta$ for which $F_\beta$ score correlates
best with human ratings.

We use an aritificial dataset, which for every sentence contains:
- the number of edits that must be performed for every sentence,
- the number of edits proposed by a model,
- the number of correct edits proposed by a model,
- human rating of the sentence.

Using bootstrap resampling, compute the mean human rating and $F_\beta$ score
for each sampled dataset and then manually compute the Pearson correlation
for betas between 0 and 2, and return the most correlating beta.

#### Tests Start: metric_correlation_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 metric_correlation.py --bootstrap_samples=100 --data_size=1000`
```
Best correlation of 0.711 was found for beta 0.79
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/metric_correlation_1.svgz)
- `python3 metric_correlation.py --bootstrap_samples=100 --data_size=2000`
```
Best correlation of 0.726 was found for beta 0.63
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/metric_correlation_2.svgz)
- `python3 metric_correlation.py --bootstrap_samples=200 --data_size=2000`
```
Best correlation of 0.676 was found for beta 0.61
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/metric_correlation_3.svgz)
#### Tests End:
```


### Assignment: decision_tree
#### Date: Deadline: Dec 12, 7:59 a.m.
#### Points: 4 points
#### Tests: decision_tree_tests

Starting with the [decision_tree.py](https://github.com/ufal/npfl129/tree/master/labs/09/decision_tree.py),
manually implement construction of a classification decision tree, supporting both
`gini` and `entropy` criteria, and `max_depth`, `min_to_split` and `max_leaves`
constraints.

#### Tests Start: decision_tree_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 decision_tree.py --dataset=digits --criterion=gini --min_to_split=250`
```
Train accuracy: 60.7%
Test accuracy: 59.6%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_1.svgz)
- `python3 decision_tree.py --dataset=digits --criterion=gini --max_depth=3`
```
Train accuracy: 41.1%
Test accuracy: 38.0%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_2.svgz)
- `python3 decision_tree.py --dataset=digits --criterion=gini --max_leaves=8`
```
Train accuracy: 60.1%
Test accuracy: 57.1%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_3.svgz)
- `python3 decision_tree.py --dataset=digits --criterion=gini --min_to_split=220 --max_leaves=8`
```
Train accuracy: 60.7%
Test accuracy: 59.6%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_4.svgz)
- `python3 decision_tree.py --dataset=digits --criterion=entropy --min_to_split=420`
```
Train accuracy: 42.4%
Test accuracy: 40.2%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_5.svgz)
- `python3 decision_tree.py --dataset=breast_cancer --criterion=entropy --max_depth=3 --seed=44`
```
Train accuracy: 94.8%
Test accuracy: 93.7%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_6.svgz)
- `python3 decision_tree.py --dataset=digits --criterion=entropy --max_leaves=7`
```
Train accuracy: 53.2%
Test accuracy: 51.6%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_7.svgz)
- `python3 decision_tree.py --dataset=breast_cancer --criterion=entropy --min_to_split=55 --max_depth=3 --seed=44`
```
Train accuracy: 94.4%
Test accuracy: 93.7%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/decision_tree_8.svgz)
#### Tests End:


```
### Assignment: random_forest
#### Date: Deadline: Dec 12, 7:59 a.m.
#### Points: 3 points
#### Examples: random_forest_examples
#### Tests: random_forest_tests

Using the [random_forest.py](https://github.com/ufal/npfl129/tree/master/labs/09/random_forest.py)
template, train a random forest, which is a collection of decision trees trained
with dataset bagging and random feature subsampling.

#### Examples Start: random_forest_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 random_forest.py --dataset=wine --trees=3 --max_depth=3`
```
Train accuracy: 99.2%
Test accuracy: 88.9%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/random_forest_1.svgz)
- `python3 random_forest.py --dataset=wine --trees=3 --bagging --max_depth=3`
```
Train accuracy: 97.7%
Test accuracy: 95.6%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/random_forest_2.svgz)
- `python3 random_forest.py --dataset=wine --trees=3 --feature_subsampling=0.5 --max_depth=3`
```
Train accuracy: 97.7%
Test accuracy: 88.9%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/random_forest_3.svgz)
- `python3 random_forest.py --dataset=wine --trees=3 --bagging --feature_subsampling=0.5 --max_depth=3`
```
Train accuracy: 99.2%
Test accuracy: 95.6%
```
![Example visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2223/tasks/figures/random_forest_4.svgz)
#### Examples End:
#### Tests Start: random_forest_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 random_forest.py --dataset=digits --trees=10 --max_depth=3`
```
Train accuracy: 54.4%
Test accuracy: 50.4%
```
- `python3 random_forest.py --dataset=digits --trees=10 --bagging --max_depth=3`
```
Train accuracy: 72.8%
Test accuracy: 72.2%
```
- `python3 random_forest.py --dataset=digits --trees=10 --feature_subsampling=0.5 --max_depth=3`
```
Train accuracy: 64.3%
Test accuracy: 62.7%
```
- `python3 random_forest.py --dataset=digits --trees=10 --bagging --feature_subsampling=0.5 --max_depth=3`
```
Train accuracy: 73.5%
Test accuracy: 75.6%
```
- `python3 random_forest.py --dataset=wine --trees=10 --max_depth=3`
```
Train accuracy: 99.2%
Test accuracy: 88.9%
```
- `python3 random_forest.py --dataset=wine --trees=10 --bagging --max_depth=3`
```
Train accuracy: 100.0%
Test accuracy: 97.8%
```
- `python3 random_forest.py --dataset=breast_cancer --trees=10 --feature_subsampling=0.5 --max_depth=3`
```
Train accuracy: 97.9%
Test accuracy: 95.1%
```
- `python3 random_forest.py --dataset=breast_cancer --trees=10 --bagging --feature_subsampling=0.5 --max_depth=3`
```
Train accuracy: 98.6%
Test accuracy: 95.1%
```
#### Tests End:
```
