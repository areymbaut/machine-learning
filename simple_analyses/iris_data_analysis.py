
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt


# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target
feature_names, label_names = iris.feature_names, iris.target_names.tolist()
label_names = [x.capitalize() for x in label_names]
n_samples, n_features = X.shape

# Visualize data
plt.rcParams["figure.figsize"] = [7.00, 5.5]
plt.rcParams["figure.autolayout"] = True
fontsize = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 3])
ax.set_xlabel(feature_names[0].capitalize(), fontsize=fontsize)
ax.set_ylabel(feature_names[1].capitalize(), fontsize=fontsize)
ax.set_zlabel(feature_names[2].capitalize(), fontsize=fontsize)
ax.set_title('Iris dataset')
cb = fig.colorbar(p, ax=ax, shrink=0.5, location='left')
cb.set_label(label=feature_names[3].capitalize(), size=fontsize)
plt.show(block=False)

###############################################################################
###############################################################################
###############################################################################

# Check if data is a good candidate for PCA reduction
pca = PCA(n_components=n_features)
pca.fit(X)

plt.rcParams["figure.figsize"] = [7.00, 5.5]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
p = ax.bar(np.arange(n_features),
           pca.explained_variance_ratio_*100)
ax.bar_label(p, fmt=lambda x: f'{x:.2f} %')
plt.xticks(ticks=np.arange(n_features),
           labels=[str(i+1) for i in np.arange(n_features, dtype=np.int32)])
plt.yticks(ticks=np.arange(0, 105, 20))
plt.ylim([0, 105])
plt.xlabel('Principal components')
plt.ylabel('Percentage of variance')
plt.title('PCA decomposition')
plt.show(block=False)

# Run PCA dimensionality reduction with 2 components
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
explained_var = pca.explained_variance_ratio_*100

plt.rcParams["figure.figsize"] = [7.00, 5.5]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
ax.legend(scatter.legend_elements()[0], label_names)
ax.set_xlabel(rf'PC$_1$ ({explained_var[0]:.2f} % of the variance)')
ax.set_ylabel(rf'PC$_2$ ({explained_var[1]:.2f} % of the variance)')
plt.title('PCA decomposition')
plt.show(block=False)

###############################################################################
###############################################################################
###############################################################################

# Split dataset into training/testing datasets
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=42)

# Optimize hyper-parameters using cross-validation
pipeline = Pipeline(steps=[
    ("pca", PCA(n_components=2)),
    ("scaler", StandardScaler()),
    ("model", SVC(kernel='rbf')),
])
parameter_grid = [
    {'model__C': [0.5, 1, 5, 10, 100],
     'model__gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001]}
]
model_grid = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)
model_grid.fit(X_train, y_train)
print('Optimal radial SVM parameters: ', model_grid.best_params_)

# Optimal pipeline
pipeline = Pipeline(steps=[
    ("pca", PCA(n_components=2)),
    ("scaler", StandardScaler()),
    ("model", SVC(kernel='rbf',
                  C=model_grid.best_params_['model__C'],
                  gamma=model_grid.best_params_['model__gamma'])),
])
pipeline.fit(X_train, y_train)

# Test model
y_pred = pipeline.predict(X_test)
accuracy = np.sum(y_pred == y_test)/len(y_test)*100
print(f'Accuracy of radial SVM = {accuracy:.2f} %')

confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(
    confusion_mat,
    display_labels=label_names)
plt.rcParams["figure.figsize"] = [9.00, 6]
plt.rcParams["figure.autolayout"] = True
disp.plot()
plt.show(block=False)

###############################################################################
###############################################################################
###############################################################################

# Visualize
_, pca_pipeline = pipeline.steps[0]
_, scaler_pipeline = pipeline.steps[1]
_, svm_pipeline = pipeline.steps[2]

X_train_pca = pca_pipeline.transform(X_train)
X_test_pca = pca_pipeline.transform(X_test)

# Generate decision surface over dense grid
X_full = np.row_stack((X_train_pca, X_test_pca))
x_1 = X_full[:, 0]
x_2 = X_full[:, 1]
x_min = x_1.min() - 1
x_max = x_1.max() + 1
y_min = x_2.min() - 1
y_max = x_2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step=0.1),
                     np.arange(y_min, y_max, step=0.1))
X_pca_grid = np.column_stack((xx.ravel(), yy.ravel()))
z_pred = svm_pipeline.predict(scaler_pipeline.transform(X_pca_grid))
z_pred = z_pred.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10,10))
ax.contourf(xx, yy, z_pred, alpha=0.1)
scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
                     s=100, marker='o', edgecolors='k', alpha=0.2,
                     label = 'Training data')
scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test,
                     s=100, marker='^', edgecolors='k', label = 'Testing data')
ax.set_xlabel(rf'PC$_1$ ({explained_var[0]:.2f} % of the variance)')
ax.set_ylabel(rf'PC$_2$ ({explained_var[1]:.2f} % of the variance)')
ax.set_title('Decision surface of radial SVM using PCA projections')
plt.legend()
plt.show()
