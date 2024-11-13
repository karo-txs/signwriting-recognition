from abc import ABC, abstractmethod
from tqdm import tqdm
import requests
import zipfile
import kaggle
import shutil
import os


class Dataset(ABC):
    """
    Abstract base class for datasets that provides a framework for downloading and mapping dataset classes.

    Attributes:
        base_path (str): Path where the dataset files will be stored.
        name (str): Name of the dataset, should be defined in each subclass.

    Methods:
        download_from_kaggle(data_name: str): Downloads and unzips the dataset from Kaggle.
        download_from_url(url: str, ignore_exists=False): Downloads a dataset from a URL.
        download(): Abstract method for downloading the dataset, to be implemented in subclasses.
        get_mapper(): Abstract method for returning the mapping dictionary, to be implemented in subclasses.
        map_classes(): Abstract method for mapping classes to specific formats, to be implemented in subclasses.
        map_classes_to_sign_writing_format(source_dir: str, target_dir: str): Maps classes to the SignWriting format.
        map_classes_to_sign_writing_format_file_name_based(source_dir: str, target_dir: str): Maps classes using filename-based matching.
    """

    def __init__(self, base_path="../assets/raw_data/"):
        """
        Initializes the Dataset with a base path for saving files.

        Args:
            base_path (str): Path where dataset files will be saved. Defaults to '../assets/raw_data/'.
        """
        self.base_path = base_path

    def get_base_path(self):
        """
        Returns the base path for saving dataset files.

        Returns:
            str: The base path where dataset files are saved.
        """
        return self.base_path

    def download_from_kaggle(self, data_name: str):
        """
        Downloads a dataset from Kaggle and extracts it into the specified base path.

        Args:
            data_name (str): The Kaggle dataset identifier in the format 'user/dataset-name'.
        """
        dataset_path = f'{self.get_base_path()}/{self.name}/original'
        if not os.path.isdir(dataset_path):
            kaggle.api.dataset_download_files(
                data_name, path=dataset_path, unzip=True)

    def download_from_url(self, url: str, ignore_exists=False):
        """
        Downloads a dataset from a specified URL and extracts it.

        Args:
            url (str): URL to download the dataset from.
            ignore_exists (bool): If True, downloads the dataset even if it already exists locally. Defaults to False.
        """
        dataset_path = f'{self.get_base_path()}/{self.name}/original'
        if ignore_exists or not os.path.isdir(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            file_type = url.split(".")[-1]

            # Make a HEAD request to get file size
            response = requests.head(url)
            total_size = int(response.headers.get('content-length', 0))

            # Download the file
            with requests.get(url, stream=True) as r, open(f'{dataset_path}/file.{file_type}', 'wb') as file, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=f'{dataset_path}/file.{file_type}', ascii=True
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            # Extract the file if it is a zip file
            with zipfile.ZipFile(f'{dataset_path}/file.{file_type}', 'r') as zip_ref:
                zip_ref.extractall(dataset_path)

    @abstractmethod
    def download(self):
        """Abstract method for downloading the dataset. Should be implemented in the subclass."""
        pass

    @abstractmethod
    def get_mapper(self):
        """Abstract method to return the mapping dictionary for class names. Should be implemented in the subclass."""
        pass

    @abstractmethod
    def map_classes(self):
        """Abstract method to map dataset classes. Should be implemented in the subclass."""
        pass

    def map_classes_to_sign_writing_format(self, source_dir: str, target_dir: str):
        """
        Maps dataset classes to the SignWriting format by copying files from the source directory to the target directory.

        Args:
            source_dir (str): Directory containing the original files for each class.
            target_dir (str): Directory where the files should be copied in SignWriting format.
        """
        if os.path.isdir(target_dir):
            print(f"The folder {target_dir} has already been mapped")
            return

        for old_name, new_name in self.get_mapper().items():
            print(f"Copying: {old_name} to {new_name}")

            target_path = os.path.join(target_dir, new_name)
            os.makedirs(target_path, exist_ok=True)

            old_path = os.path.join(source_dir, old_name)
            if os.path.exists(old_path):
                for file_name in os.listdir(old_path):
                    old_file_path = os.path.join(old_path, file_name)
                    new_file_path = os.path.join(target_path, file_name)
                    shutil.copy2(old_file_path, new_file_path)

    def map_classes_to_sign_writing_format_file_name_based(self, source_dir: str, target_dir: str):
        """
        Maps classes using filename-based matching for datasets where files are named according to class labels.

        Args:
            source_dir (str): Directory containing the original files.
            target_dir (str): Directory where the files should be copied.
        """
        if os.path.isdir(target_dir):
            print(f"The folder {target_dir} has already been mapped")
            return

        for file_name in os.listdir(source_dir):
            for old_name, new_name in self.get_mapper().items():
                if file_name.startswith(old_name):
                    target_path = os.path.join(target_dir, new_name)
                    os.makedirs(target_path, exist_ok=True)

                    old_file_path = os.path.join(source_dir, file_name)
                    new_file_path = os.path.join(target_path, file_name)
                    print(f"Copying {file_name} to {target_path}")
                    shutil.copy2(old_file_path, new_file_path)
                    break
