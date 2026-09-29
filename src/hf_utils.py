import os
from pathlib import Path
from huggingface_hub import HfApi, Repository, hf_hub_url, cached_download, create_repo, upload_file
from huggingface_hub.utils import HfHubHTTPError

def ensure_repo_exists(repo_id: str, token: str = None, private: bool = False):
    """
    Create the repository on HF Hub if it does not exist.
    """
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, token=token)
        # Repo exists
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            # Repo does not exist, create it
            create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        else:
            raise

def upload_file_to_hf(local_path: str, repo_id: str, path_in_repo: str = None, token: str = None):
    """
    Upload a single file to the HF repo.
    If path_in_repo is None, uses the file name.
    """
    api = HfApi()
    if path_in_repo is None:
        path_in_repo = os.path.basename(local_path)
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
    )
    return f"https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}"

def download_file_from_hf(repo_id: str, filename: str, token: str = None, cache_dir: str = None):
    """
    Download a file from the HF repo and return local path.
    """
    return cached_download(
        hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model"),
        token=token,
        cache_dir=cache_dir,
    )

def download_folder_from_hf(repo_id: str, repo_type: str = "model", token: str = None, cache_dir: str = None):
    """
    Clone the whole repo (like git lfs) using Repository.
    Returns local path.
    """
    local_dir = os.path.join(cache_dir or ".", repo_id.replace("/", "_"))
    if not os.path.exists(local_dir):
        Repository(
            local_dir=local_dir,
            clone_from=repo_id,
            token=token,
        )
    else:
        # Update
        repo = Repository(local_dir=local_dir, clone_from=repo_id, token=token)
        repo.git_pull()
    return local_dir