import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import neighbors, metrics

data = pd.read_csv('winequality-white.csv', sep=";")

# Apercu du dataset:
'''
print(data.head())
print(data.describe())
print(data.shape)
print(list(data.columns.values))
'''
# X contient les points(multidim) et y contient les etiquettes (dim 1,note de qualité)
X = data[data.columns[:-1]].values
y = data['quality']

# print('X shape is:',X.shape,'y shape is:',y.shape)

'''
# Affichage sous forme d'histogramme pour chacune des variables
fig = plt.figure(figsize=(16, 12)
# on parcours l'espace des colonnes de X
for feat_idx in range(X.shape[1]):
    # ajout des graphs (vides)
    ax=fig.add_subplot(3, 4, feat_idx + 1)
    # définition desdits graphs
    # Histogrammes pour l'ensemble des valeurs de chaque colonnes
    h=ax.hist(X[:, feat_idx], bins=50, color='steelblue', density=True, edgecolor='none')
    # Assignation des labels de chaque colonne comme titre de chaque graph
    ax.set_title(data.columns[feat_idx], size=14)
plt.show()
'''
# on transforme le probleme en pb de classification
# les deux classes seront quality=6 et quality<6
y_class = np.where(y < 6, 0, 1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=0.3)

# Onstandardise les données d'entrainement et de test
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

'''
# Affichage des données standardisées
fig = plt.figure(figsize=(16, 12))
for feat_idx in range(X_train_std.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))
    h = ax.hist(X_train_std[:, feat_idx], bins=50, color = 'steelblue', density=True, edgecolor='none')
    ax.set_title(data.columns[feat_idx], fontsize=14)
plt.show()
'''

# Fixer les valeurs des hyperparamètres à tester
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19]}

# Choisir un score à optimiser, ici l'accuracy (proportion de prédictions correctes)
score = 'accuracy'

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
clf = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(),  # un classifieur kNN
    param_grid,     # hyperparamètres à tester
    cv=5,           # nombre de folds de validation croisée
    scoring=score   # score à optimiser
)

# Optimiser ce classifieur sur le jeu d'entraînement
clf.fit(X_train_std, y_train)

# Afficher le(s) hyperparamètre(s) optimaux
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print(clf.best_params_)

# Afficher les performances correspondantes
print("Résultats de la validation croisée :")
for mean, std, params in zip(
    clf.cv_results_['mean_test_score'],  # score moyen
    clf.cv_results_['std_test_score'],  # écart-type du score
    clf.cv_results_['params']           # valeur de l'hyperparamètre
):

    print("{} = {:.3f} (+/-{:.03f}) for {}".format(
        score,
        mean,
        std * 2,
        params
    ))


y_pred = clf.predict(X_test_std)
print("\nSur le jeu de test, l'accuracy est: {:.3f}".format(metrics.accuracy_score(y_test, y_pred)))

