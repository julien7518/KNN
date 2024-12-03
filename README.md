
# Rapport de code - KNN
Moustafa Adamaly & Julien Fernandes

## Introduction

Ce rapport documente le fonctionnement du code implémentant un algorithme de classification **k-Nearest Neighbors (k-NN)**. Il inclut la gestion des distances, la normalisation des données, et les processus d'importation/exportation des données. L'objectif est d'expliquer comment le code fonctionne et comment l'utiliser dans un contexte d'analyse de données.

---

## Organisation du code

Le code est organisé en deux fichiers principaux :

1. **`knn.py`** : Définit les structures de données, les distances, et l'algorithme k-NN.
2. **`__main__.py`** : Contient des utilitaires pour la gestion des données et l'exécution des tests sur l'algorithme k-NN.

---

## Contenu du module `knn.py`

### Classe `Point`

Cette classe représente un point dans un espace à 7 dimensions. Elle est définie comme suit :

```python
class Point(object):
    def __init__(self, point_id: int, coordinates: list[float], label: int | None = None) -> None:
```

- **Attributs** :
  - `id` : Identifiant unique du point.
  - `coordinates` : Liste des coordonnées dans l'espace 7D.
  - `label` : (Optionnel) Classe associée au point.

- **Validation** : Une exception `DimensionError` est levée si le nombre de dimensions est incorrect.

### Fonctions de distances

Deux fonctions principales calculent les distances entre deux points :

1. **Distance de Minkowski** :
   $$
   d(p_1, p_2) = \left( \sum_{i=1}^{n} |p_1[i] - p_2[i]|^p \right)^{1/p}
   $$
   - Si $p = 1$, on obtient la distance Manhattan.
   - Si $p = 2$, on obtient la distance Euclidienne.

2. **Distance de Tchebychev** :
   $$
   d(p_1, p_2) = \max_i |p_1[i] - p_2[i]|
   $$

### Algorithme k-NN

La fonction `knn` applique l'algorithme k-Nearest Neighbors :

```python
def knn(k: int,
        new: Point,
        dataset: list[Point],
        fct_distance: Callable[[Point, Point], float]
       ) -> Tuple[int, int]:
```

- **Entrées** :
  - `k` : Nombre de voisins à considérer.
  - `new` : Le point à classer.
  - `dataset` : Ensemble de points de référence.
  - `fct_distance` : Fonction de distance à utiliser.

- **Sortie** :
  - Une prédiction sous la forme $(id, label)$.

- **Étapes principales** :
  1. Calcul des distances entre le point cible et les points du dataset.
  2. Tri des distances.
  3. Attribution du label majoritaire parmi les $k$ plus proches voisins.

---

## Contenu du module `__main__.py`

### Importation des données

La fonction `import_csv` convertit un fichier CSV en une liste d'instances `Point`.

```python
def import_csv(name: str,
               full_data: bool = True
              ) -> list[Point]:
```

- Les colonnes sont interprétées comme suit :
  - `id` : Identifiant.
  - Coordonnées : Colonnes $1 \text{ \`a\ } n-1$.
  - Label (optionnel) : Dernière colonne.

### Normalisation

1. **Min-Max Scaling** :
   $$
   x^* = \frac{x - \min(x)}{\max(x) - \min(x)}
   $$
   Implémenté par `min_max_scaler`.

2. **Standardisation** :
   $$
   x^* = \frac{x - \mu}{\sigma}
   $$
   Où $\mu$ est la moyenne et $\sigma$ est l'écart-type, implémenté par `std_mean_normalization`.

### Création d'un dataset de test

La fonction `create_dataset` sépare les données en deux ensembles :

- **Training** : 80% des données.
- **Test** : 20% des données restantes.

```python
def create_dataset(dataset: list[Point]) -> list[list[Point]]:
```

### Évaluation de la performance

La fonction `fitness` évalue la précision du modèle sur un ensemble de test :

```python
def fitness(dataset: list[Point],
            testing_set: list[Point],
            k_knn: int,
            d: Callable[[Point, Point], float]
           ) -> float:
```

- **Retourne** : Le pourcentage de prédictions correctes $\displaystyle \frac{\text{correct}}{\text{total}}$.

### Exportation des résultats

Les résultats peuvent être sauvegardés dans un fichier CSV via `export_result_csv`.

---

## Utilisation

### Étapes principales

1. **Importer les données** :

   ```python
   datas = import_csv("data\train.csv")
   ```

2. **Normaliser les données** (optionnel) :

   ```python
   min_max_scaler(datas)
   ```

3. **Créer un ensemble de test et d'entraînement** :

   ```python
   train_set, test_set = create_dataset(datas)
   ```

4. **Exécuter le k-NN** :

   ```python
   result = knn(k=3, new=new_point, dataset=train_set, fct_distance=euclidean)
   ```

5. **Évaluer le modèle** :

   ```python
   accuracy = fitness(train_set, test_set, k_knn=3, d=euclidean)
   ```

6. **Exporter les résultats** :

   ```python
   export_result_csv(result, "output.csv")
   ```

---

## Résultats

Les résultats des tests réalisés sont présentés dans le tableau suivant, regroupés par normalisation, nombre de voisins $k$, et fonction de distance utilisée. Les scores moyens indiquent la performance globale des différentes configurations testées.

| Normalisation         | $k$ | Distance              | Score Moyen |
|-----------------------|-------|-----------------------|-------------|
| Min-Max Scaler        | 1     | Minkowski (p=2)      | 0.996       |
| Min-Max Scaler        | 3     | Tchebychev      | 0.862       |
| Std-Mean Normalization| 1     | Minkowski (p=1)      | 0.978       |
| Std-Mean Normalization| 2     | Minkowski (p=2)      | 0.974       |
| Aucune                | 1     | Minkowski (p=1)      | 0.985       |
| Aucune                | 2     | Tchebychev      | 0.837       |
| Aucune                | 1     | Minkowski (p=2)      | 0.983       |

**Description des tests :**  
Seuls certaines configurations, qui nous semblait être les plus interésantes, sont retranscrites dans le tableau. Nous avons délibérément omis certains tests afin de conserver un tableau lisible et concis.

- **Processeur utilisé :** Intel64 Family 6 Model 140 Stepping 1
- **Taille de l'ensemble d'entraînement :** 809 points (80% de train.csv)
- **Taille de l'ensemble de test :** 203 points (20% restant de train.csv)
- **Temps d'exécution moyen :** 0.24 secondes par test

## Conclusion

Les tests réalisés montrent que les meilleures performances ont été obtenues avec une normalisation des données (Min-Max Scaler ou Std-Mean Normalization), particulièrement pour la distance euclidienne ($p=2$) et $k=1$. En revanche, des valeurs de $k \geq 3$ ont donné de moins bons résultats. La distance de Tchebychev et la distance de Minkowski avec des valeurs de $p$ non standard (autres que 1 ou 2) se sont également révélées inadaptées. Cependant, sur la plateforme Kaggle, les meilleurs résultats ont été obtenus avec $k=1$ et la distance de Manhattan, sans normalisation. Ces conclusions sont spécifiques à ce jeu de données et ne peuvent pas être généralisées.
