import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import KI.timing


data = pd.read_csv("csv/dataset_405_altholz_5classes.csv")
data = data.iloc[:, 1:]
le = LabelEncoder()
sc = StandardScaler()
data['art'] = le.fit_transform(data["art"])
data[["intensity_std", "intensity_mean", "phase_std", "phase_mean", "mod_std", "mod_mean"]] = sc.fit_transform(
    data[["intensity_std", "intensity_mean", "phase_std", "phase_mean", "mod_std", "mod_mean"]])
Y = data.iloc[:, 0]
X = data.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1, shuffle=True)

print("X_Train Shape: ", x_train.shape)
print("X_Test Shape: ", x_test.shape)
print("Y_Train Shape: ", y_train.shape)
print("Y_Test Shape: ", y_test.shape)

kf = KFold(n_splits=5, shuffle=True)

mlp = MLPClassifier(max_iter=500, random_state=1, verbose=True, activation='relu', alpha=0.0001,
                    hidden_layer_sizes=(256, 128, 64), learning_rate='constant', solver='adam', shuffle=True)

# parameter_space = {
#     'hidden_layer_sizes': [(512, 256, 128), (512, 256, 128, 64), (256, 128, 64), (256, 128, 64, 32)],
#     'activation': ['relu', 'tanh', 'identity', 'logistic'],
#     'solver': ['sgd', 'adam', 'lbfgs'],
#     'alpha': [0.0001, 0.05, 0.00001],
#     'learning_rate': ['constant', 'adaptive'],
# }
#
# clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
# grid_search = clf.fit(x_train, y_train)
#
# print('Best parameters found:\n', clf.best_params_)
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
# enc = OneHotEncoder()
# Y = np.array(Y)
# Y = Y.reshape(len(Y), 1)
# Y = enc.fit_transform(Y)
#
# result = cross_val_score(mlp, X, Y, cv=kf)
# print('Avg accuracy: {}'.format(result.mean()))

mlp.fit(x_train, y_train)


def accuracy(confucion_matrix):
    diagonal_sum = confucion_matrix.trace()
    sum_of_all_elements = confucion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


y_pred = mlp.predict(x_test)
cm = confusion_matrix(y_pred, y_test)
print("Accuracy of MLPClassifier: ", accuracy(cm))
class_names = ["Ahorn", "Buche", "Eiche", "Fichte", "Kiefer", "Laerche"]
guete = ["AI", "AII", "AIII", "AIV", "Praep"]
plot_confusion_matrix(mlp, x_test, y_test, display_labels=guete)
plt.show()

print("done")


