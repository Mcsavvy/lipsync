import os
import shutil
import subprocess

from easy_functions import load_file_from_url, load_model, load_predictor
import warnings
from os import path

# Get the location of the basicsr package

version = "v8.3"

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)


# Get the location of the basicsr package
def get_basicsr_location():
    result = subprocess.run(["pip", "show", "basicsr"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "Location: " in line:
            loc = line.split("Location: ")[1]
            if loc.endswith("site-packages") or loc.endswith("dist-packages"):
                loc = path.join(loc, "basicsr")
            if loc.endswith("basicsr"):
                loc = path.join(loc, "data")
            return loc
    return None


# Move and replace a file to the basicsr location
def move_and_replace_file_to_basicsr(file_name):
    basicsr_location = get_basicsr_location()
    if basicsr_location:
        destination = path.join(basicsr_location, file_name)
        # Move and replace the file
        shutil.copyfile(file_name, destination)
        print(f"File replaced at {destination}")
    else:
        print("Could not find basicsr location.")


# Example usage
file_to_replace = "degradations.py"  # Replace with your file name
move_and_replace_file_to_basicsr(file_to_replace)


from enhance import load_sr

working_directory = os.getcwd()


if path.exists("installed.txt"):
    print("Already installed!")
    print("If you want to reinstall, delete the installed.txt file")
    exit(0)

has_previous_installation = input(
    "Have you previously installed this app? (y/n): "
).strip().lower() in ["y", "yes"]

if has_previous_installation:
    installation_path: str | None = None
    while installation_path is None:
        try:
            installation_path = input(
                "Enter the path where you installed the app: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("Exiting...")
            exit(0)
        if not path.exists(installation_path):
            print("Invalid path. Try again.")
            installation_path = None
        if not path.isdir(installation_path):
            print("Path is not a directory. Try again.")
            installation_path = None
        if not path.exists(path.join(installation_path, "invoke.py")):
            print("Path does not contain the app. Try again.")
            installation_path = None
        installation_path = path.abspath(installation_path)
    if path.exists(path.join(installation_path, "checkpoints")):
        print(f"Copying models from {installation_path}")
        shutil.copytree(
            path.join(installation_path, "checkpoints"),
            path.join(working_directory, "checkpoints"),
            dirs_exist_ok=True,
        )
    if path.exists(path.join(installation_path, "face_alignment")):
        print(f"Copying face alignment from {installation_path}")
        shutil.copytree(
            path.join(installation_path, "face_alignment"),
            path.join(working_directory, "face_alignment"),
            dirs_exist_ok=True,
        )
    if path.exists(path.join(installation_path, "gfpgan")):
        print(f"Copying gfpgan from {installation_path}")
        shutil.copytree(
            path.join(installation_path, "gfpgan"),
            path.join(working_directory, "gfpgan"),
            dirs_exist_ok=True,
        )
    if path.exists(path.join(installation_path, "last_detected_face.pkl")):
        print(f"Copying last detected face from {installation_path}")
        shutil.copyfile(
            path.join(installation_path, "last_detected_face.pkl"),
            path.join(working_directory, "last_detected_face.pkl"),
        )

# create essential directories
os.makedirs(path.join(working_directory, "checkpoints"), exist_ok=True)
os.makedirs(path.join(working_directory, "temp"), exist_ok=True)
os.makedirs(path.join(working_directory, "face_alignment"), exist_ok=True)
os.makedirs(path.join(working_directory, "result"), exist_ok=True)

# download mobilenet model
if not path.exists(path.join(working_directory, "checkpoints", "mobilenet.pth")):
    print("downloading mobilenet essentials")
    load_file_from_url(
        url="https://github.com/anothermartz/Easy-Wav2Lip/raw/v8.3/checkpoints/mobilenet.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="mobilenet.pth",
    )

# download and initialize both wav2lip models
if not path.exists(path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pk1")):
    print("downloading wav2lip_gan essentials")
    load_file_from_url(
        url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="Wav2Lip_GAN.pth",
    )
    load_model(path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pth"))
    print("wav2lip_gan loaded")

if not path.exists(path.join(working_directory, "checkpoints", "Wav2Lip.pk1")):
    print("downloading wav2lip essentials")
    load_file_from_url(
        url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="Wav2Lip.pth",
    )
    load_model(path.join(working_directory, "checkpoints", "Wav2Lip.pth"))
    print("wav2lip loaded")

# download gfpgan files
if not path.exists(path.join(working_directory, "checkpoints", "GFPGANv1.4.pth")):
    print("downloading gfpgan essentials")
    load_file_from_url(
        url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/GFPGANv1.4.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="GFPGANv1.4.pth",
    )
load_sr()

# load face detectors
if not path.exists(
    path.join(
        working_directory, "checkpoints", "shape_predictor_68_face_landmarks_GTX.dat"
    )
):
    print("initializing face detectors")
    load_file_from_url(
        url="https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/shape_predictor_68_face_landmarks_GTX.dat",
        model_dir="checkpoints",
        progress=True,
        file_name="shape_predictor_68_face_landmarks_GTX.dat",
    )
load_predictor()

# write a file to signify setup is done
with open("installed.txt", "w") as f:
    f.write(version)
print("Installation complete!")
