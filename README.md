# MLOps TP1 - Churn Baseline

Ce projet met en place une baseline de machine learning pour predire le churn client.
Il illustre les bonnes pratiques MLOps de base : organisation du projet, suivi des
experiences avec MLflow, versionnement du code avec Git et suivi des donnees avec DVC.

## Structure du projet

```text
data/
  raw/           Donnees brutes non transformees
  processed/     Donnees post-transformation
src/              Code source
artifacts/        Sorties d'entrainement, modeles et metriques
notebooks/        Exploration
airflow/
  dags/           DAGs Airflow
  logs/           Logs d'execution Airflow
  plugins/        Plugins Airflow
docker-compose.yml Environnement Airflow + PostgreSQL
README.md         Documentation du projet
requirements.txt  Dependances Python
.gitignore        Fichiers ignores par Git
```

## Installation

```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'entrainement

```powershell
python main.py
```

Le script lit les donnees depuis `data/raw/churn.csv` et ecrit les sorties dans
`artifacts/`.

Pendant l'execution, le script enregistre aussi un run MLflow avec :

- les parametres du modele ;
- les metriques d'evaluation ;
- les artefacts generes ;
- le modele sklearn dans le MLflow Model Registry sous le nom
  `churn-classifier`.

Le modele publie recoit l'alias `latest-cindy`, ce qui permet de le recharger sans
reentrainer :

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-classifier@latest-cindy")
print(type(model))
```

Comme le modele est entraine avec un DataFrame pandas, scikit-learn conserve les
colonnes d'entree dans `model.feature_names_in_`. A l'inference, l'API s'en sert
pour realigner les features :

```python
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
```

## Visualiser les experiences MLflow

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

L'experience utilisee par le script s'appelle `churn-baseline`.

## Orchestration avec Airflow

Le pipeline est aussi orchestre avec Airflow dans le DAG :

```text
airflow/dags/churn_pipeline.py
```

Le DAG contient les taches suivantes :

```text
check_data >> prepare_data >> train_model >> evaluate_model >> save_artifacts
```

Role des taches :

- `check_data` verifie que le dataset existe.
- `prepare_data` prepare les jeux d'entrainement et de test.
- `train_model` entraine le modele.
- `evaluate_model` calcule les metriques.
- `save_artifacts` sauvegarde le modele et les metriques.

Lancer Airflow :

```powershell
docker compose up
```

Puis ouvrir :

```text
http://localhost:8080
```

Identifiants locaux :

```text
admin / admin
```

Le conteneur Airflow monte les dossiers du projet dans `/opt/airflow/project` :

```text
src/       -> /opt/airflow/project/src
data/      -> /opt/airflow/project/data
artifacts/ -> /opt/airflow/project/artifacts
```

## Passage de donnees entre taches Airflow

Le DAG utilise XCom pour transmettre uniquement de petites informations entre les
taches, comme des chemins de fichiers ou un dictionnaire de metriques.

Les objets volumineux ne sont pas passes via XCom :

```text
DataFrames -> data/processed/
Modele     -> artifacts/model.pkl
Metriques  -> artifacts/metrics.txt
Chemins    -> XCom
```

Cette approche evite de surcharger la base metadata Airflow.

## Controle d'erreur dans Airflow

La tache `check_data` detecte rapidement l'absence du dataset. Par exemple, si
`data/raw/churn.csv` est renomme en `data/raw/churnn.csv`, le DAG echoue des la
premiere tache avec une erreur explicite :

```text
FileNotFoundError: Dataset introuvable : /opt/airflow/project/data/raw/churn.csv
```

Ce controle permet d'arreter le pipeline avant la preparation, l'entrainement ou
l'evaluation lorsque la donnee attendue est absente.

## Suivi des donnees avec DVC

Le fichier de donnees `data/raw/churn.csv` est suivi avec DVC afin d'eviter de
versionner directement les donnees brutes dans Git.

Pour ajouter une donnee a DVC :

```powershell
dvc add data/raw/churn.csv
git add data/raw/churn.csv.dvc data/raw/.gitignore
```

Pour recuperer les donnees suivies par DVC apres un clone :

```powershell
dvc pull
```

## Synthese : role de chaque outil

| Outil | Role principal | Question a laquelle il repond |
| --- | --- | --- |
| Git | Versionner le code et les metadonnees du projet. | Quelle version du code est utilisee ? |
| MLflow | Suivre les runs, les parametres, les metriques, les artefacts et les modeles. | Que s'est-il passe pendant cette execution ? |
| DVC | Suivre les donnees et les artefacts volumineux hors logique Git. | Quelle version de la donnee est associee au projet ? |

## Service d'inference

L'API FastAPI expose le modele de churn charge depuis le MLflow Model Registry :

```text
models:/churn-classifier@latest-cindy
```

Avant de lancer l'API, entrainer et enregistrer le modele si necessaire :

```powershell
.\env\Scripts\activate
python main.py
```

Lancer ensuite le service d'inference :

```powershell
uvicorn app:app --port 8001
```

La documentation Swagger est disponible a l'adresse :

```text
http://127.0.0.1:8001/docs
```

Format d'entree attendu par `/predict` :

```json
{
  "age": 35,
  "tenure_months": 12,
  "monthly_charges": 75.5,
  "support_calls": 2,
  "contract_type": "Monthly",
  "payment_method": "Electronic check",
  "internet_service": "Fiber"
}
```

Exemple d'appel avec `curl` :

```cmd
curl.exe -X POST "http://127.0.0.1:8001/predict" -H "Content-Type: application/json" -d "{\"age\":35,\"tenure_months\":12,\"monthly_charges\":75.5,\"support_calls\":2,\"contract_type\":\"Monthly\",\"payment_method\":\"Electronic check\",\"internet_service\":\"Fiber\"}"
```

Exemple de reponse :

```json
{
  "prediction": 0,
  "label": "no_churn",
  "confidence": 0.87
}
```

## Supervision

La route `/metrics` expose des indicateurs simples conserves en memoire :

```cmd
curl.exe "http://127.0.0.1:8001/metrics"
```

Indicateurs disponibles :

- `n_requests` : nombre de requetes recues sur `/predict`.
- `n_errors` : nombre d'erreurs serveur pendant la prediction.
- `n_invalid_inputs` : nombre d'entrees invalides ou anormales.
- `drift_alerts` : nombre d'alertes d'anomalie ou de derive detectees.
- `predictions_distribution` : repartition des predictions `churn` et `no_churn`.
- `evidently_reports_generated` : nombre de rapports Evidently generes, si au moins un rapport a ete cree.

Les erreurs de format sont gerees par Pydantic/FastAPI avec un statut HTTP `422`
et incrementent `n_invalid_inputs`. Exemple : champ manquant ou type incorrect.

Les anomalies metier sont des valeurs valides en type, mais hors des plages
observees a l'entrainement. Par exemple, `tenure_months=999` retourne :

```json
{
  "prediction": null,
  "warning": "anomalie detectee",
  "details": ["tenure_months=999 hors plage [1, 72]"]
}
```

Dans ce cas, l'API log un `WARNING`, incremente `n_invalid_inputs` et
`drift_alerts`, et ne lance pas la prediction.

La derive temps reel est detectee avec une fenetre glissante sur les dernieres
requetes. Si la moyenne recente s'eloigne trop de la moyenne d'entrainement,
l'API log un `WARNING` et incremente `drift_alerts`.

Les logs applicatifs sont ecrits dans :

```text
logs/api.log
```

Pour consulter les dernieres lignes sous Windows :

```cmd
powershell -Command "Get-Content logs\api.log -Tail 30"
```

## Rapports de derive

La route `/drift_report` genere un rapport Evidently HTML a partir des 100
dernieres requetes valides traitees par `/predict`.

Prerequis :

- `artifacts/reference_sample.csv` doit exister.
- Le buffer doit contenir 100 predictions recentes.

Le fichier de reference est genere pendant l'entrainement par `python main.py`.

Pour remplir le buffer avec 100 requetes valides :

```cmd
for /L %i in (1,1,100) do curl.exe -s -X POST "http://127.0.0.1:8001/predict" -H "Content-Type: application/json" -d "{\"age\":35,\"tenure_months\":12,\"monthly_charges\":75.5,\"support_calls\":2,\"contract_type\":\"Monthly\",\"payment_method\":\"Electronic check\",\"internet_service\":\"Fiber\"}"
```

Generer ensuite le rapport :

```cmd
curl.exe -X POST "http://127.0.0.1:8001/drift_report"
```

Exemple de reponse :

```json
{
  "status": "report_generated",
  "report_path": "logs/drift_reports/drift_20260617_122902.html"
}
```

Les rapports HTML sont sauvegardes dans :

```text
logs/drift_reports/
```

Pour lister les rapports :

```cmd
dir logs\drift_reports
```

Pour ouvrir un rapport dans le navigateur :

```cmd
start logs\drift_reports\drift_20260617_122902.html
```

Dans le rapport Evidently, les colonnes avec derive indiquent que la distribution
des donnees recentes differe de l'echantillon de reference. Les variables
numeriques sont comparees avec des tests statistiques adaptes, et les variables
categorielles avec des tests de distribution. Ce rapport sert a l'analyse
approfondie, tandis que la detection maison de `/predict` sert a l'alerte rapide.

## CI/CD

Le projet utilise GitHub Actions pour executer une pipeline de CI definie dans :

```text
.github/workflows/ci.yml
```

La CI se declenche automatiquement :

- a chaque `push` sur les branches `main` et `develop` ;
- a chaque `pull_request` vers `main`.

Verifications automatisees :

- recuperation du code avec `actions/checkout` ;
- installation de Python 3.11 avec `actions/setup-python` ;
- installation des dependances depuis `requirements.txt` ;
- creation d'un petit jeu de donnees de test si la donnee brute n'est pas presente ;
- generation des artefacts avec `python main.py` ;
- verification de la presence de `artifacts/model.pkl` et `artifacts/reference_sample.csv` ;
- execution des tests avec `pytest tests/ -v` ;
- demarrage de l'API avec `uvicorn` ;
- verification de la route `/health` avec `curl`.

Pour lancer les tests en local :

```cmd
.\env\Scripts\python.exe -m pytest tests/ -v
```

Ou, si l'environnement virtuel est deja active :

```cmd
pytest tests/ -v
```

La CI sert a verifier qu'une modification ne casse pas le projet : preparation des
donnees, API, tests de validation et test de non-regression du modele.

Le CD correspondrait aux etapes de mise a disposition d'une nouvelle version apres
validation par la CI. Pour ce projet, on pourrait ajouter :

- publier une image Docker de l'API sur un registry, par exemple Docker Hub ou GitHub Container Registry ;
- deployer automatiquement l'API mise a jour sur un environnement de staging ;
- publier le modele entraine dans MLflow Model Registry avec un tag ou alias `staging` ;
- declencher un redeploiement automatique apres promotion du modele ou de l'image en production.

Ces etapes ne sont pas implementees dans ce TP. Elles representent la suite logique :
la CI valide la qualite, puis le CD prepare ou effectue le deploiement controle de
la nouvelle version.
