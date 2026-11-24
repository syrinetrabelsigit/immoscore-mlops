# ImmoScore MLOps

Pipeline reproductible de prédiction du prix des maisons (jeu de données
Kaggle *House Prices*), réalisé dans le cadre du cours MLOps
(chapitre 2 : reproductibilité).

## Objectif

Transformer un notebook exploratoire en un projet reproductible : un
collègue, sur une autre machine et à plusieurs mois d'intervalle, doit
obtenir le même modèle et les mêmes métriques à partir du même code et des
mêmes données.

## Ce que fait le projet

1. **Préparation** : conserve les variables numériques les plus corrélées
   au prix (`SalePrice`), selon un seuil défini dans `params.yaml`.
2. **Découpage** : sépare les données en jeux d'entraînement et de test
   (stratifié, graine fixée).
3. **Entraînement** : régression linéaire (scikit-learn).
4. **Évaluation** : RMSE et R² sur train et test, écrits dans
   `reports/metrics.json`.

## Outils

| Besoin | Outil |
|---|---|
| Versionner le code | Git, GitHub |
| Paramètres et graine | `params.yaml` |
| Versionner les données | DVC, stockage S3 (Supabase) |
| Enchaîner les étapes | `dvc.yaml` |
| Isoler l'environnement | Docker |

## Structure du projet

```
immoscore-mlops/
├── data/            données (versionnées avec DVC, ignorées par Git)
├── models/          modèles produits (ignorés par Git)
├── notebooks/       exploration uniquement
├── reports/         métriques produites
├── src/             scripts : data_process, split, train, eval
├── params.yaml      paramètres et graine aléatoire
├── dvc.yaml         déclaration du pipeline
├── requirements.txt dépendances Python
├── Dockerfile       image du projet
└── README.md
```

## Installation

```bash
git clone https://github.com/<utilisateur>/immoscore-mlops.git
cd immoscore-mlops
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash : source .venv/Scripts/activate
pip install -r requirements.txt "dvc[s3]"
```

Les données ne sont pas dans Git. Pour les récupérer, il faut les clés du
stockage distant (à demander au Team Lead, jamais dans le dépôt) :

```bash
dvc remote modify --local s3 access_key_id '<cle>'
dvc remote modify --local s3 secret_access_key '<secret>'
dvc pull
```

## Utilisation

Exécuter tout le pipeline :

```bash
dvc repro
```

Ou avec Docker :

```bash
docker build -t immoscore:v1 .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" immoscore:v1
```

## Équipe

- **Syrine Trabelsi** : Team Lead
- **Cyrine Sammouda** : Membre