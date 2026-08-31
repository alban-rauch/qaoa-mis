from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="your-username/qaoa-mis-results",
    repo_type="dataset",
    local_dir="data/dataset",
)
