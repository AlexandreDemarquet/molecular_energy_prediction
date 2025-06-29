## Structure du projet

### Notebooks principaux

* **`dataAnalyse.ipynb`**
  Analyse exploratoire du dataset :

  * Visualisation d'exemples de molécules
  * Statistiques sur la distribution des données

* **`histogramme.ipynb`**
  Régression basée sur les histogrammes des types d’atomes présents dans chaque molécule.

* **`spectreMatriceCoulomb.ipynb`**
  Utilisation du spectre de la matrice de Coulomb pour prédire l’énergie des molécules.

* **`contrastive.ipynb`**
  Méthode de contrastive learning pour apprendre des représentations invariantes aux rotations/translations,
  suivie d’une régression sur ces représentations.

* **`regression_scattering_et_ou_contrastive.ipynb`**
  Prédiction de l’énergie à partir :

  * des coefficients de scattering
  * et/ou des embeddings issus du contrastive learning
    Utilisation de modèles de régression multi-linéaire.

* **`visualisation_scattering.ipynb`**
  Visualisation des lobes obtenus par convolution de la densité avec des filtres de scattering.

---

### Dossiers

* **`dataset/`**
  Scripts de chargement des données :

  * `QM7XDataset.py` : chargement du dataset QM7
  * `UnlabeledContrastive.py` : dataset non étiqueté pour le contrastive learning

* **`data/`**
  Contient les fichiers de données :

  * Jeux de données d'entraînement et de test 

* **`pkl/`**
  Fichiers enregistrés avec Pickle (modèles, embeddings, représentations, etc.)

* **`rapport/`**
  Images du rapport LaTeX (visualisations, figures, graphiques, etc.)

* **`models/`**
  Modèles utilisés dans le projet :

  * `contrastive_learning.py` : encodeur pour le contrastive learning
  * `regression_multi_lin.py` : modèle de régression multi-linéaire

* **`README.md`**
  Présentation générale du projet et de sa structure.
