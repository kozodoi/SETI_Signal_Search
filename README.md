# SETI E.T. Signal Search

Top-7% solution to the [SETI Breakthrough Listen](https://www.kaggle.com/c/seti-breakthrough-listen) Kaggle competition on the E.T. signal detection.

![sample](https://i.postimg.cc/Kztyq0Lg/seti-sample.jpg)


## Summary

Estimating text complexity and readability is a crucial task for school teachers. Offering students text passages at the right level of challenge is important for facilitating a fast development of reading skills. The existing tools to estimate text complexity rely on weak proxies and heuristics, which results in a suboptimal accuracy. This project uses deep learning to predict the readability scores of text passages.

My solution is an ensemble of eight transformer models, including BERT, RoBERTa and others. All transformers are implemented in `PyTorch` and feature a custom regression head that uses a concatenated output of multiple hidden layers. The modeling pipeline implements text augmentations such as sentence order shuffle, backtranslation and injecting target noise. The table below summarizes the main architecture and training parameters. The solution places in the top-9% of the Kaggle competition leaderboard.


## Project structure

The project has the following structure:
- `codes/`: `.py` main scripts with data, model, training and inference modules
- `notebooks/`: `.ipynb` Colab-friendly notebooks with model training and blending
- `input/`: input data not included due to size limits and can be downloaded [here](https://www.kaggle.com/c/seti-breakthrough-listen/data)
- `output/`: model configurations, predictions and figures exported from the notebooks


## Working with the repo

### Environment

To work with the repo, I recommend to create a virtual Conda environment from the `environment.yml` file:
```
conda env create --name seti --file environment.yml
conda activate seti
```

### Reproducing solution

The solution can be reproduced in the following steps:
1. Download input data and place it in the `input/` folder.
1. Run training notebooks `training_v01.ipynb` and `training_v02.ipynb` to obtain model predictions.
3. Run the ensembling notebook `blending.ipynb` to obtain the final predictions.

More details are provided in the documentation within the scripts & notebooks.
