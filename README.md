# MLOps TP1 - Churn Baseline

Projet de baseline ML pour predire le churn client.

## Structure du projet

```text
data/
  raw/          Donnees brutes non transformees
  processed/    Donnees post-transformation
src/             Code source
artifacts/       Sorties d'entrainement, modeles et metriques
notebooks/       Exploration
```

## Installation

```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'entrainement

```powershell
python src/baseline_train.py
```

Le script lit les donnees depuis `data/raw/churn.csv` et ecrit les sorties dans `artifacts/`.
