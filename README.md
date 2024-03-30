# machine-learning

This repository gathers code ressources related to machine learning.

## Reimplementations
The `reimplementations` directory contains scripts that reimplement basic machine-learning algorithms and neural networks from scratch (_i.e._, using non-machine-learning-specific libraries such as `numpy` or using basic functions from `torch`). Each reimplementation is accompanied by a short demonstration on simple datasets. Here are the algorithms that have been reimplemented:
- linear and logistic regressions.
- Naive Bayes classifier.
- _k_-means classifier.
- _k_ nearest neighbors (classifier and regressor).
- linear support-vector-machine (SVM) classifier.
- decision tree (classifier and regressor).
- random forest (classifier and regressor).
- AdaBoost classifier.
- gradient-boosted trees (classifier and regressor).
- single-layer perceptron classifier.
- linear regression (using `torch`).
- long short-term memory (LSTM) neural network (using `torch`).

## Simple analyses
The `simple_analyses` directory contains scripts that implement data-processing pipelines involving various aspects of data pre-processing, model training, inference and data interpretation. Example scripts include:
- `digits_pca_umap_kmeans_svm.py` | Dimensionality reduction using PCA and UMAP, _k_-means clustering and radial SVM classification optimized via cross-validation (using `sklearn`).
- `heart_disease_decision_tree_pruning.py` | Heart-disease classification of patients using decision trees and optimization via cost-complexity pruning and cross-validation (using `sklearn`).
- `images_convolutional_neural_network.py` | Image classification via a simple convolution neural network (using `torch`)
