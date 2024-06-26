"""Common functions for the project."""

import os
from io import BytesIO
from pathlib import Path

import dropbox  # type: ignore[import]
from dotenv import load_dotenv

load_dotenv()


def get_dropbox_client() -> dropbox.Dropbox:
    """Get a Dropbox client object"""
    access_token = os.environ["DBX_ACCESS_TOKEN"]
    app_key = os.environ["DBX_APP_KEY"]
    app_secret = os.environ["DBX_APP_SECRET"]

    return dropbox.Dropbox(
        oauth2_access_token=access_token, app_key=app_key, app_secret=app_secret
    )


def download_wav2lip_model(client: dropbox.Dropbox):
    """Download Wav2Lip model from Dropbox"""
    model_file_path = os.environ["WAV2LIP_MODEL_FILE_PATH"]
    metadata, res = client.files_download(model_file_path)
    return BytesIO(res.content)


def download_wav2lip_gan_model(client: dropbox.Dropbox):
    """Download Wav2Lip GAN model from Dropbox"""
    model_file_path = os.environ["WAV2LIP_GAN_MODEL_FILE_PATH"]
    metadata, res = client.files_download(model_file_path)
    return BytesIO(res.content)

def ensure_wav2lip_model(client: dropbox.Dropbox):
    """Ensure that the Wav2Lip model is downloaded"""
    model_path = Path("checkpoints/wav2lip.pth")
    if not model_path.exists():
        model = download_wav2lip_model(client)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(model.read())

def ensure_wav2lip_gan_model(client: dropbox.Dropbox):
    """Ensure that the Wav2Lip GAN model is downloaded"""
    model_path = Path("checkpoints/wav2lip_gan.pth")
    if not model_path.exists():
        model = download_wav2lip_gan_model(client)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(model.read())

def ensure_models(client: dropbox.Dropbox):
    """Ensure that the Wav2Lip models are downloaded"""
    ensure_wav2lip_model(client)
    ensure_wav2lip_gan_model(client)
