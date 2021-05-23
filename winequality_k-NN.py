import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('winequality-white.csv', sep=";")

# Apercu du dataset:
print(data.head())
print(data.describe())
print(data.shape)
print(list(data.columns.values))

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