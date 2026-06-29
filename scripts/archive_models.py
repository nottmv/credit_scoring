import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
name = "credit_scoring_model"
versions = client.search_model_versions(f"name='{name}'")
if not versions:
    print("No versions found.")
else:
    latest = max(int(v.version) for v in versions)
    for v in versions:
        if int(v.version) != latest:
            client.transition_model_version_stage(name=name, version=v.version, stage="Archived")
            print(f"Archived version {v.version}")
    print(f"Version {latest} remains in Staging.")
