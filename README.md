# SETI E.T. Signal Search

Top-7% solution to the [SETI Breakthrough Listen](https://www.kaggle.com/c/seti-breakthrough-listen) Kaggle competition on the E.T. signal detection.

![sample](https://i.postimg.cc/Kztyq0Lg/seti-sample.jpg)


## Summary

Searching for extraterrestrial signals from deep space is one of the main tasks of The Breakthrough Listen team at the University of California, Berkeley. Current methods compare scans of the target stars with scans of other regions of sky to detect anomalous signals. However, standard detection techniques might miss signals with complex time or frequency structure, and those in regions of the spectrum with lots of interference.

This project uses deep learning to detect extraterrestrial signals in radio spectrograms. The modeling pipeline involves converting radio spectrograms into 2D images and employing computer vision models. My solution is a blend of two models: EfficientNet B7 and SWIN Transformer. Both models are implemented in `PyTorch`. The solution places in the top-7% of the Kaggle competition leaderboard.


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
