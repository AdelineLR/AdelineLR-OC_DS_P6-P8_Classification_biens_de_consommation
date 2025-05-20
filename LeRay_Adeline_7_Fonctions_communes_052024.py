#!/usr/bin/env python
# coding: utf-8

# Projet P6 - Segmentez des clients d'un site e-commerce : Fonctions
# OPENCLASSROOMS - Parcours Data Scientist - Adeline Le Ray - 05/2024


import numpy as np
import pandas as pd

# graphiques
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster, metrics, manifold, decomposition
import time

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, silhouette_score


# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters
def ARI_fct(features, 
            l_cat, 
            y_cat_num, 
            perplexity=30, 
            init_tsne='random', 
            max_iter=2000, 
            learning_rate=200, 
            n_init=10,
            init_kmeans='k-means++',
            random_state=42):
    """
    @brief Calcul du t-SNE, détermination des clusters et calcul de l'ARI entre les vraies catégories et les n° de clusters.
    
    @param features: Les caractéristiques à utiliser pour le t-SNE (numpy array ou pandas DataFrame).
    @param l_cat: Liste des catégories (list).
    @param y_cat_num: Les vraies catégories sous forme numérique (numpy array).
    @param perplexity: Perplexité pour t-SNE (int).
    @param init_tsne : Initialisation pour t-SNE, 'random' ou 'pca'(string).
    @param max_iter: Nombre maximum d'itérations pour t-SNE (int).
    @param learning_rate: Taux d'apprentissage pour t-SNE (int).
    @param n_init: Nombre d'initialisations pour KMeans (int).
    @param init_kmeans : Méthode d'initialisation pour KMeans (str).
    @return ARI: L'indice de Rand ajusté (float).
    @return X_tsne: Les coordonnées t-SNE (numpy array).
    @return cls.labels_: Les labels des clusters (numpy array).
    """
    time1 = time.time()
    num_labels = len(l_cat)
    tsne = manifold.TSNE(n_components=2, 
                         perplexity=perplexity, 
                         init=init_tsne, 
                         max_iter=max_iter, 
                         random_state=random_state)
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après t-SNE 
    cls = cluster.KMeans(n_clusters=num_labels, n_init=n_init, init=init_kmeans, random_state=random_state)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_), 4)
    time2 = np.round(time.time() - time1, 0)
    print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_


# visualisation du t-SNE selon les vraies catégories et selon les clusters
def TSNE_visu_fct(X_tsne, y_cat_num, labels, ARI, l_cat, figsize=(15, 6), cmap='Set1'):
    """
    @brief Visualisation du t-SNE selon les vraies catégories et selon les clusters.
    
    @param X_tsne: Les coordonnées t-SNE (numpy array).
    @param y_cat_num: Les vraies catégories (list ou numpy array).
    @param labels: Les labels des clusters (list ou numpy array).
    @param ARI: L'indice de Rand ajusté (float).
    @param l_cat: Liste des catégories (list).
    @param figsize: Taille de la figure (tuple).
    @param cmap: Colormap à utiliser pour le scatter plot (str).
    """
    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cat_num, cmap=cmap)

    handles, _ = scatter.legend_elements()
    legend_labels = [str(cat) for cat in l_cat]
    
    ax.legend(handles, 
              legend_labels,
              title="Categorie",
              bbox_to_anchor=(1.0, 1), 
              loc=2, 
              borderaxespad=0)
    ax.set_title('Représentation des produits par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=cmap)
    ax.legend(handles=scatter.legend_elements()[0], 
              labels=[f'Cluster {i}' for i in range(len(set(labels)))],
              title="Clusters",
              bbox_to_anchor=(1.0, 1), 
              loc=2, 
              borderaxespad=0)
    plt.title('Représentation des produits par clusters')
    
    plt.tight_layout()
    plt.show()    


def reordered_confusion_matrix(y_true, y_predict, class_names):
    """!
    @brief Calcule et visualise la matrice de confusion réordonnée.

    Cette fonction prend les vraies étiquettes et les étiquettes prédites, puis calcule et visualise la matrice 
    de confusion réordonnée en utilisant un algorithme d'affectation linéaire pour résoudre le problème d'assignation.

    @param y_true: Les vraies étiquettes (array ou list).
    @param y_predict: Les étiquettes prédites par le modèle (array ou list).
    @param class_names: Liste des catégories (list).
    """
    # Calculer la matrice de confusion initiale
    cm = confusion_matrix(y_true, y_predict)
  
    # Calculer la matrice de coût
    s = np.max(cm)
    cost_matrix = -cm + s

    # Résoudre le problème d'assignation
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Réorganiser la matrice de confusion basée sur l'assignation optimale
    cm2 = cm[:, col_ind]

    # Réordonner les noms des classes
    reordered_class_names = [class_names[i] for i in col_ind]

    # Afficher la matrice de confusion réorganisée 
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=reordered_class_names, rotation=0)
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=col_ind, rotation=0)
    plt.xlabel('Predictions')
    plt.ylabel('True category')
    plt.title("Matrice de Confusion")
    plt.show()
    
    return cm2

def model_score(eval_df, model, ARI, X_tsne, labels, confusion_matrix, l_cat):
    """!
    @brief Met à jour eval_df avec les scores de précision pour un modèle donné.
    
    @param eval_df: DataFrame d'évaluation à mettre à jour (DataFrame).
    @param model: Nom du modèle (str).
    @param ARI: Adjusted Rand Index du modèle (float).
    @param X_tsne: Coordonnées t-SNE des données après réduction de dimension (numpy array).
    @param labels: Étiquettes de clusters ou de classes obtenues par le modèle (numpy array).
    @param confusion_matrix: Matrice de confusion (numpy array).
    @param l_cat: Liste des catégories (list).
    
    @return eval_df: DataFrame d'évaluation mis à jour (DataFrame).
    """
    # Calcul de la précision par class
    precision_dict = {cat: calculate_precision_for_class(confusion_matrix, i) for i, cat in enumerate(l_cat)}
    
    # Calcul du recall par class
    recall_dict = {cat: calculate_recall_for_class(confusion_matrix, i) for i, cat in enumerate(l_cat)}
    
    # Nouvelle ligne
    new_row = {'Model': model, 'ARI': ARI, 'silhouette_score': round(silhouette_score(X_tsne, labels),2)}
    for cat in precision_dict:
        precision = precision_dict[cat]
        recall = recall_dict[cat]
        new_row.update({f'Precision / recall - {cat}': "{:.2f} / {:.2f}".format(precision, recall)})
    new_row = pd.DataFrame(new_row, index=[0])
    
    # Ajouter la nouvelle ligne à eval_df
    return pd.concat([eval_df, new_row],ignore_index=True)


def calculate_recall_for_class(confusion_matrix, class_index):
    """!
    @brief Calcule la précision pour une classe donnée à partir d'une matrice de confusion.
    
    @param confusion_matrix: Matrice de confusion (numpy array).
    @param class_index: Index de la classe pour laquelle la précision doit être calculée (int).
    
    @return recall: Recall pour la classe spécifiée (float).
    """
    # Vrais Positifs (TP) et Faux Positifs (FP)
    TP = confusion_matrix[class_index, class_index]
    FN = np.sum(confusion_matrix[class_index,:]) - TP
    
    # Calcul du recall
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
    
    return recall


def calculate_precision_for_class(confusion_matrix, class_index):
    """!
    @brief Calcule la précision pour une classe donnée à partir d'une matrice de confusion.
    
    @param confusion_matrix: Matrice de confusion (numpy array).
    @param class_index: Index de la classe pour laquelle la précision doit être calculée (int).
    
    @return precision: Précision pour la classe spécifiée (float).
    """
    # Vrais Positifs (TP) et Faux Positifs (FP)
    TP = confusion_matrix[class_index, class_index]
    FP = np.sum(confusion_matrix[:, class_index]) - TP
    
    # Calcul de la précision
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0
    
    return precision


# Analyse des sous-catégories dans les clusters
def plot_category_proportions(df, category, ax):
    """!
    @brief Crée et affiche un graphique à barres empilées horizontales montrant la proportion de chaque cluster dans chaque
    sous-catégorie pour une catégorie spécifiée.

    Cette fonction prend un DataFrame contenant des données sur les clusters et une catégorie spécifiée. Elle calcule 
    la proportion de chaque cluster dans chaque sous-catégorie pour la catégorie donnée, puis crée et affiche un graphique 
    à barres empilées horizontales pour visualiser ces proportions.

    @param df: DataFrame contenant les données sur les clusters et les catégories (DataFrame).
    @param category: Catégorie spécifiée pour laquelle les proportions de clusters sont calculées (str).
    @param ax: Axes object où le graphique sera dessiné.

    @return ax: Axes object représentant le graphique.
    """
    # Calculer le nombre d'occurrences de chaque sous-catégorie dans chaque cluster
    df_category = df[df['main_category'] == category]
    counts = df_category.groupby(['sub_category', 'cluster'], as_index=False).agg(count=('cluster', 'size'))
    total = counts.groupby('sub_category', as_index=False).agg(total_count=('count', 'sum'))
    counts = counts.merge(total, on='sub_category')
    counts['proportion'] = counts['count'] / counts['total_count']
    pivot_df = counts.pivot(index='sub_category', columns='cluster', values='proportion').fillna(0)

    # Afficher le barplot
    pivot_df.plot(kind='barh', stacked=True, ax=ax)
    ax.set_title(f'Proportion de chaque cluster dans chaque sous-catégorie pour {category}')
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Sous-catégorie')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    return ax


def TSNE_visu_subcat_fct(X_tsne, 
                         y_cat_num, 
                         y_subcat_num_by_category, 
                         subcategories_by_category, 
                         l_cat, 
                         df, 
                         figsize=(15, 6), 
                         cmap='Set1'):
    """
    @brief Visualisation du t-SNE selon les sous-catégories par catégorie, avec le diagramme à barres des proportions 
    des clusters.
    
    @param X_tsne: Les coordonnées t-SNE (numpy array).
    @param y_cat_num: Numéros des catégories pour chaque point (numpy array).
    @param y_subcat_num_by_category: Liste des numéros de sous-catégories par catégorie (list de list ou numpy array).
    @param subcategories_by_category: Liste des noms de sous-catégories par catégorie (list de list).
    @param l_cat: Liste des catégories (list).
    @param df: DataFrame contenant les données sur les clusters et les catégories (DataFrame).
    @param figsize: Taille de la figure (tuple).
    @param cmap: Colormap à utiliser pour le scatter plot (str).
    """
    # Déterminer le nombre de lignes et de colonnes pour la grille de sous-figures
    nb_rows = len(l_cat)
    nb_cols = 2
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=figsize, tight_layout=True)

    # Déterminer les limites des axes pour une représentation similaire de chaque catégorie
    x_min, x_max = np.min(X_tsne[:, 0]) - 5, np.max(X_tsne[:, 0]) + 5
    y_min, y_max = np.min(X_tsne[:, 1]) - 5, np.max(X_tsne[:, 1]) + 5
    
    for i, category in enumerate(l_cat):
        # Filtrer X_tsne pour la catégorie actuelle
        X_tsne_cat = X_tsne[np.array(y_cat_num) == i]
        sub_cat_labels = subcategories_by_category[i]
        subcat_nums = y_subcat_num_by_category[i]

        # Plot t-SNE
        ax_tsne = axes[i, 0]
        scatter = ax_tsne.scatter(X_tsne_cat[:, 0], X_tsne_cat[:, 1], c=subcat_nums, cmap=cmap)
        ax_tsne.set_title(f"Catégorie : {category}")
        ax_tsne.set_xlim(x_min, x_max)
        ax_tsne.set_ylim(y_min, y_max)

        # Ajout de la légende pour le t-SNE
        ax_tsne.legend(handles=scatter.legend_elements()[0], 
                      labels=sub_cat_labels, 
                      title="Sous-catégorie",
                      bbox_to_anchor=(1.05, 1), 
                      loc='upper left',
                      borderaxespad=0)
        
        # Plot du diagramme à barres
        ax_bar = axes[i, 1]
        plot_category_proportions(df, category, ax=ax_bar)
    
    plt.show()