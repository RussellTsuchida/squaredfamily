# squaredfamily
Code to accompany the paper: "Squared families: Searching beyond regular probability models"


`model_fit.py` fits a squared family model to some data, as in the paper. It computes 500 maximum likelihood estimates for different dataset samples from the same ground truth target density.

`visualise.py` shows the parameter estimate distributions compared with the asymptotic estimates derived in the paper. Run this after `model_fit.py`.

By default, running both files sequentially reproduces the seed 0 result. The
seed can be changed on line 283 of `model_fit.py`. The number of samples `N`
can be changed on line 280 of `model_fit.py` and line 25 of `visualise.py`.
