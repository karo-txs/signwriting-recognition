from typing import Iterable, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import kaggle
import shutil
import os


@dataclass
class Dataset(ABC):
    base_path: str = field(default="assets/raw/")
    name: str = field(default=None)

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
        dataset_path = f"{self.get_base_path()}/{self.name}/original"
        if not os.path.isdir(dataset_path):
            print(
                "Baixando dataset do kaggle, esse processo pode demorar algums minutos."
            )
            os.makedirs(dataset_path, exist_ok=True)
            kaggle.api.dataset_download_files(
                data_name, path=dataset_path, unzip=True, quiet=False
            )

    def download_zenodo_record(self, record_id: int, token=None):
        api_url = f"https://zenodo.org/api/records/{record_id}/files-archive"

        dataset_path = f"{self.get_base_path()}/{self.name}/original"
        if not os.path.isdir(dataset_path):
            print(
                "Baixando dataset do zenodo, esse processo pode demorar algums minutos."
            )
            os.makedirs(dataset_path, exist_ok=True)

            with requests.get(api_url, stream=True) as r, open(
                f"{dataset_path}/dataset.zip", "wb"
            ) as fp:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)

    def download_from_url(self, url: str, ignore_exists=False):
        """
        Downloads a dataset from a specified URL and extracts it.

        Args:
            url (str): URL to download the dataset from.
            ignore_exists (bool): If True, downloads the dataset even if it already exists locally. Defaults to False.
        """
        dataset_path = f"{self.get_base_path()}/{self.name}/original"
        if ignore_exists or not os.path.isdir(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            file_type = url.split(".")[-1]

            # Make a HEAD request to get file size
            response = requests.head(url)
            total_size = int(response.headers.get("content-length", 0))

            # Download the file
            with requests.get(url, stream=True) as r, open(
                f"{dataset_path}/file.{file_type}", "wb"
            ) as file, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"{dataset_path}/file.{file_type}",
                ascii=True,
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            # Extract the file if it is a zip file
            with zipfile.ZipFile(f"{dataset_path}/file.{file_type}", "r") as zip_ref:
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

    def map_classes_to_sign_writing_format(
        self,
        source_dir: str,
        target_dir: str,
        exclude_prefix: Optional[str] = None,
    ):
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
                    if exclude_prefix and file_name.startswith(exclude_prefix):
                        continue

                    old_file_path = os.path.join(old_path, file_name)
                    new_file_path = os.path.join(target_path, file_name)
                    shutil.copy2(old_file_path, new_file_path)

    def map_classes_to_sign_writing_format_file_name_based(
        self,
        source_dir: Union[str, Path, Iterable[Union[str, Path]]],
        target_dir: Union[str, Path],
        *,
        recursive: bool = False,
        include_extensions: Optional[Tuple[str, ...]] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".tiff",
            ".webp",
        ),
        exclude_extensions: Optional[Tuple[str, ...]] = (".depth.png", ".iseg.png"),
        not_ignore_exists: bool = True,
    ):
        """
        Copia arquivos para pastas‑classe novas, renomeando-as conforme
        `self.get_mapper()`.

        Parâmetros principais
        ---------------------
        source_dirs        : Caminho único ou coleção de caminhos de origem.
        target_dir         : Diretório‑raiz de destino.
        recursive          : Se True, percorre subdiretórios recursivamente.
        include_extensions : Apenas arquivos com essas extensões são copiados.
                            `None` → ignora esse filtro (aceita tudo).
        exclude_extensions : Extensões a descartar (sobrepõe include).
                            `None` → não exclui nada.
        not_ignore_exists  : Se True, aborta se `target_dir` já existir.
        """
        target_dir = Path(target_dir)

        if target_dir.is_dir() and not_ignore_exists:
            print(f"The folder {target_dir} has already been mapped")
            return

        if isinstance(source_dir, (str, Path)):
            sources = [Path(source_dir)]
        else:
            sources = [Path(p) for p in source_dir]

        mapper = self.get_mapper()

        def _is_valid_file(fname: str) -> bool:
            if exclude_extensions and fname.endswith(exclude_extensions):
                return False
            if include_extensions:
                return fname.lower().endswith(include_extensions)
            return True

        for src_root in sources:
            if not src_root.is_dir():
                print(f"[AVISO] {src_root} não é diretório; ignorando.")
                continue

            files_iter = (
                (p for p in src_root.rglob("*") if p.is_file())
                if recursive
                else (src_root / f for f in os.listdir(src_root))
            )

            for src_path in files_iter:
                file_name = src_path.name

                if not _is_valid_file(file_name):
                    continue

                for old_name, new_name in mapper.items():
                    if file_name.startswith(old_name):
                        dst_class_dir = target_dir / new_name
                        dst_class_dir.mkdir(parents=True, exist_ok=True)

                        dst_path = dst_class_dir / file_name
                        print(f"Copying {src_path} → {dst_path}")
                        shutil.copy2(src_path, dst_path)
                        break
