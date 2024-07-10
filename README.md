An MLOps implementation on GCP using Vertex AI, BigQuery, and TensorFlow for a predictive Customer Lifetime Value (CLV) use case.

## Objectives

* Train a TensorFlow model locally in a hosted [**Vertex Notebook**](https://cloud.google.com/vertex-ai/docs/general/notebooks?hl=sv).
* Create a [**managed Tabular dataset**](https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets?hl=sv) artifact for experiment tracking.
* Containerize your training code with [**Cloud Build**](https://cloud.google.com/build) and push it to [**Google Cloud Artifact Registry**](https://cloud.google.com/artifact-registry).
* Run a [**Vertex AI custom training job**](https://cloud.google.com/vertex-ai/docs/training/custom-training) with your custom model container.
* Use [**Vertex TensorBoard**](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview) to visualize model performance.
* Deploy your trained model to a [**Vertex Online Prediction Endpoint**](https://cloud.google.com/vertex-ai/docs/predictions/getting-predictions) for serving predictions.
* Request an online prediction and explanation and see the response.

## Setup and Execution steps

### 1. Use `gcloud` to enable the services:

```
gcloud services enable \
  compute.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  notebooks.googleapis.com \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com
```

### 2. Create Vertex AI custom service account for Vertex Tensorboard experiment tracking

#### 2.1. Create custom service account
```
SERVICE_ACCOUNT_ID=vertex-custom-training-sa
gcloud iam service-accounts create $SERVICE_ACCOUNT_ID  \
    --description="A custom service account for Vertex custom training with Tensorboard" \
    --display-name="Vertex AI Custom Training"
```

#### 2.2. Grant it access to GCS for writing and retrieving Tensorboard logs
```
PROJECT_ID=$(gcloud config get-value core/project)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SERVICE_ACCOUNT_ID@$PROJECT_ID.iam.gserviceaccount.com \
    --role="roles/storage.admin"
```

#### 2.3. Grant it access to your BigQuery data source to read data into your TensorFlow model
```
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SERVICE_ACCOUNT_ID@$PROJECT_ID.iam.gserviceaccount.com \
    --role="roles/bigquery.admin"
```

#### 2.4. Grant it access to Vertex AI for running model training, deployment, and explanation jobs
```
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SERVICE_ACCOUNT_ID@$PROJECT_ID.iam.gserviceaccount.com \
    --role="roles/aiplatform.user"
```

### 3. Clone the repository

In your **JupyterLab** instance, open a terminal and clone this repository in the `home` folder.
```
cd
git clone https://github.com/manoharkaranth/vertexai-mlops-predictive-clv.git
```

### 4. Install the dependencies

Pip install `requirements.txt` to install dependencies:

```bash
cd vertexai-mlops-predictive-clv
pip install -U -r requirements.txt
```

### 5. Execution of the project

Execute `lab_exercise_long.ipynb` to complete the project. 
