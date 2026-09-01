from huggingface_hub import HfApi

api = HfApi()

api.create_repo(repo_id="alban-rauch/qaoa-mis-results", repo_type="dataset", private=False)

api.upload_large_folder(
    repo_id="alban-rauch/qaoa-mis-results",
    repo_type="dataset",
    folder_path="data/dataset",
)

