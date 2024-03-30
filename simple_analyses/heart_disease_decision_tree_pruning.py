
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import pandas as pd
import ucimlrepo
import matplotlib.pyplot as plt


# Load data
df = ucimlrepo.fetch_ucirepo(id=45).data.original

# Name of label columns and names of categorical
# columns that will require one-hot encoding
label_column = 'num'
category_names = ['cp', 'restecg', 'slope', 'thal']

# Check data
print('Data types:')
print(df.dtypes, '\n')

print('Number of NaNs:')
print(df.isna().sum(), '\n')

print('Unique target values:')
print(df[label_column].unique(), '\n')

# Take out NaNs (there are only 6 of them so no imputation needed)
df = df.dropna()

# Split data
X = df.drop(label_column, axis=1).copy()
y = df[label_column].copy()

# One-hot encoding for categorical features
X_encoded = pd.get_dummies(X, columns=category_names, dtype=np.int64)

# Binarize labels
y[y > 0] = 1
label_names = ['No heart disease', 'Heart disease']

###############################################################################
###############################################################################
###############################################################################

# Basic decision-tree classifier
X_train, X_test, y_train, y_test = train_test_split(X_encoded,
                                                    y,
                                                    random_state=42)
classifier_dt = DecisionTreeClassifier(random_state=42)
classifier_dt = classifier_dt.fit(X_train, y_train)
y_pred = classifier_dt.predict(X_test)

accuracy = np.sum(y_pred == y_test)/len(y_test)*100
print(f'Accuracy of basic decision-tree classifier = {accuracy:.2f} %')

# Show tree and confusion matrix
plt.figure(figsize=(15, 7.5))
plot_tree(classifier_dt, filled=True, rounded=True,
          class_names=label_names, feature_names=X_encoded.columns)

confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(
    confusion_mat,
    display_labels=label_names)
plt.rcParams["figure.figsize"] = [9.00, 6]
plt.rcParams["figure.autolayout"] = True
disp.plot()
plt.title('Confusion matrix of basic decision-tree classifier')
plt.show(block=False)

###############################################################################
###############################################################################
###############################################################################

# Cost-complexity pruning (ccp) using cross-validation
path = classifier_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # We ignore the max alpha,
                                   # associated with the tree root

mean_score = []
std_score = []
for alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    scores = cross_val_score(dt, X_train, y_train, cv=5)
    mean_score.append(np.mean(scores))
    std_score.append(np.std(scores))
optimal_alpha = ccp_alphas[np.argmax(mean_score)]

fig, ax = plt.subplots()
ax.errorbar(ccp_alphas, mean_score, yerr=std_score,
            fmt='o', ls='--', capsize=6)
y_lim = ax.get_ylim()
plot_range = np.diff(y_lim)[0]
ax.axvline(x=optimal_alpha, ymin=0,
           ymax=(np.max(mean_score) - y_lim[0])/plot_range,
           ls='-.', color='r', label='Optimal alpha')
ax.legend()
ax.set_xlabel('Cost-complexity pruning alpha')
ax.set_ylabel('Mean accuracy throughout cross-validation')
plt.show(block=False)

# Evaluate optimal decision tree
classifier_dt_pruned = DecisionTreeClassifier(random_state=42,
                                              ccp_alpha=optimal_alpha)
classifier_dt_pruned.fit(X_train, y_train)
y_pred = classifier_dt_pruned.predict(X_test)

accuracy = np.sum(y_pred == y_test)/len(y_test)*100
print(f'Accuracy of pruned decision-tree classifier = {accuracy:.2f} %')

# Show tree and confusion matrix
plt.figure(figsize=(15, 7.5))
plot_tree(classifier_dt_pruned, filled=True, rounded=True,
          class_names=label_names, feature_names=X_encoded.columns)

confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(
    confusion_mat,
    display_labels=label_names)
plt.rcParams["figure.figsize"] = [9.00, 6]
plt.rcParams["figure.autolayout"] = True
disp.plot()
plt.title('Confusion matrix of pruned decision-tree classifier')
plt.show()
