
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from umap import UMAP


# Load data
digits = datasets.load_digits()
X, y = digits.data, digits.target
feature_names, label_names = digits.feature_names, digits.target_names.tolist()
label_names = [str(x) for x in label_names]
n_samples, n_features = X.shape
n_digits = 10

# Visualize data
images = digits.images
fig, ax_array = plt.subplots(20, 20)
axes = ax_array.flatten()
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray_r')
plt.setp(axes, xticks=[], yticks=[], frame_on=False)
plt.tight_layout(h_pad=0.5, w_pad=0.01)
plt.show(block=False)

###############################################################################
###############################################################################
###############################################################################

# Check if data is a good candidate for PCA reduction
pca = PCA(n_components=n_features)
pca.fit(X)

n_feat_plot = 5
plt.rcParams["figure.figsize"] = [7.00, 5.5]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
p = ax.bar(np.arange(n_feat_plot),
           pca.explained_variance_ratio_[:n_feat_plot]*100)
ax.bar_label(p, fmt=lambda x: f'{x:.2f} %')
plt.xticks(ticks=np.arange(n_feat_plot),
           labels=[str(i+1) for i in np.arange(n_feat_plot, dtype=np.int32)])
plt.yticks(ticks=np.arange(0, 105, 20))
plt.ylim([0, 105])
plt.xlabel('Principal components')
plt.ylabel('Percentage of variance')
plt.title('PCA decomposition')
plt.show(block=False)

# Try UMAP instead
# (Uniform Manifold Approximation and Projection for Dimension Reduction)
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=42)

reducer = UMAP(n_neighbors=30,
               min_dist=0.0,
               n_components=2,
               n_jobs=1,
               random_state=0)
X_train_umap = reducer.fit_transform(X_train)
X_test_umap = reducer.transform(X_test)

n_clusters = n_digits
clustering = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
cluster_labels_train = clustering.fit_predict(X_train_umap)
cluster_labels_test = clustering.predict(X_test_umap)

# Associate cluster labels with digits
cluster_labels_train_2 = np.zeros_like(cluster_labels_train)
cluster_labels_test_2 = np.zeros_like(cluster_labels_test)
for i in range(n_clusters):
    idx_train = (cluster_labels_train == i)
    idx_test = (cluster_labels_test == i)

    y_cluster = y_train[idx_train]
    label_value = Counter(y_cluster).most_common(1)[0][0]  # Label mode

    cluster_labels_train_2[idx_train] = label_value
    cluster_labels_test_2[idx_test] = label_value

accuracy = np.sum(cluster_labels_test_2 == y_test)/len(y_test)*100
print(f'Accuracy of k-means clustering = {accuracy:.2f} %')

confusion_mat = confusion_matrix(y_true=y_test, y_pred=cluster_labels_test_2)
disp = ConfusionMatrixDisplay(
    confusion_mat,
    display_labels=label_names)
plt.rcParams["figure.figsize"] = [9.00, 6]
plt.rcParams["figure.autolayout"] = True
disp.plot()
plt.title('Confusion matrix of k-means classifier')
plt.show(block=False)

plt.rcParams["figure.figsize"] = [15, 5.5]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(121)
scatter = ax.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
                     c=y_train, cmap='jet', s=5)
ax.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
           c=y_test, cmap='jet', s=5)
ax.set_aspect('equal', 'datalim')
ax.legend(scatter.legend_elements()[0], label_names)
ax.set_xlabel(r'UMAP$_1$')
ax.set_ylabel(r'UMAP$_2$')
ax.set_title('UMAP data')

ax = fig.add_subplot(122)
scatter = ax.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
                     c=cluster_labels_train_2, cmap='jet', s=5, alpha=0.2)
scatter = ax.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
                     c=cluster_labels_test_2, cmap='jet', s=5, marker='^')
ax.set_aspect('equal', 'datalim')
ax.legend(scatter.legend_elements()[0], label_names)
ax.set_xlabel(r'UMAP$_1$')
ax.set_ylabel(r'UMAP$_2$')
ax.set_title(f'k-means clustering of UMAP data (k = {n_clusters})')
plt.show(block=False)

###############################################################################
###############################################################################
###############################################################################

# Optimize hyper-parameters using cross-validation
# (UMAP would take too long to run multiple times in this context,
#  so it is excluded from the pipeline)
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", SVC(kernel='rbf')),
])
parameter_grid = [
    {'model__C': [1, 10, 100, 200],
     'model__gamma': ['scale', 1, 0.1]}
]
model_grid = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)
model_grid.fit(X_train_umap, y_train)
print('Optimal radial SVM parameters: ', model_grid.best_params_)

# Optimal pipeline
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", SVC(kernel='rbf',
                  C=model_grid.best_params_['model__C'],
                  gamma=model_grid.best_params_['model__gamma'])),
])
pipeline.fit(X_train_umap, y_train)

# Test model
y_pred = pipeline.predict(X_test_umap)
accuracy = np.sum(y_pred == y_test)/len(y_test)*100
print(f'Accuracy of radial SVM = {accuracy:.2f} %')

confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(
    confusion_mat,
    display_labels=label_names)
plt.rcParams["figure.figsize"] = [9.00, 6]
plt.rcParams["figure.autolayout"] = True
disp.plot()
plt.title('Confusion matrix of radial SVM classifier')
plt.show(block=False)

###############################################################################
###############################################################################
###############################################################################

# Visualize
_, scaler_pipeline = pipeline.steps[0]
_, svm_pipeline = pipeline.steps[1]

# Generate decision surface over dense grid
X_full = np.row_stack((X_train_umap, X_test_umap))
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
ax.contourf(xx, yy, z_pred, alpha=0.1, cmap='jet')
scatter = ax.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train,
                     s=100, marker='o', edgecolors='k', alpha=0.2,
                     label = 'Training data', cmap='jet')
scatter = ax.scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=y_test,
                     s=100, marker='^', edgecolors='k',
                     label = 'Testing data', cmap='jet')
ax.set_xlabel(r'UMAP$_1$')
ax.set_ylabel(r'UMAP$_2$')
ax.set_title('Decision surface of radial SVM using UMAP projections')
plt.legend()
plt.show()
