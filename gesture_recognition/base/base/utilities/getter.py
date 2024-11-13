import importlib.resources as pkg_resources
from collections import defaultdict
from glob import glob
import os


def get_all_file_paths(root_path: str, depth=2):
    """
    The function `get_all_file_paths` retrieves all file paths within a specified root directory up to a
    certain depth level.

    Args:
        root_path (str): The `root_path` parameter is a string that represents the starting directory path
    from which you want to search for files. It is the base directory from which the search for files
    will begin.
        depth: The `depth` parameter in the `get_all_file_paths` function specifies how many levels deep
    you want to search for files starting from the `root_path`. In the function, it is used to construct
    a search pattern that includes multiple levels of directories to search for files. The default value
    for `. Defaults to 2

    Returns:
        A list of file paths within the specified root path up to the specified depth.
    """
    pattern = '/*' * depth
    search_pattern = root_path + pattern

    file_paths = [path.replace("\\", "/") for path in glob(search_pattern)]
    return file_paths


def get_all_folder_names(directory: str):
    """
    The function `get_all_folder_names` returns a list of folder names within a specified directory.

    Args:
        directory (str): A string representing the path to a directory.

    Returns:
        The function `get_all_folder_names` returns a list of folder names within the specified directory.
    """
    return [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]


def get_relation_of_files_per_folder_name(file_paths: list, folder_names: list, limit_value: int = None):
    """
    The function `get_relation_of_files_per_folder_name` organizes file paths based on folder names and
    optionally limits the number of files per folder.

    Args:
        file_paths (list): A list of file paths that you want to analyze.
        folder_names (list): Folder names is a list containing the names of folders where the files are
    located.
        limit_value (int): The `limit_value` parameter in the `get_relation_of_files_per_folder_name`
    function is used to specify a limit on the number of files that can be associated with each folder
    name. If the number of files associated with a folder name exceeds this limit, those files will be
    excluded from the result

    Returns:
        The function `get_relation_of_files_per_folder_name` returns a dictionary `relation_file_folder` 
        where the keys are file paths and the values are the names of the folders the files are located in.
    """
    # Create a mapping from folder names to their indices
    index_by_folder = dict(
        (name, index) for index, name in enumerate(folder_names))

    # Extract the folder name from each file path and map it to its corresponding index
    all_file_indices = [os.path.basename(
        "/".join(path.split("/")[:-1])) for path in file_paths]

    relation_file_folder = {}

    # Create the relation_file_folder mapping
    for origin_data_path, folder_name in zip(file_paths, all_file_indices):
        relation_file_folder[origin_data_path] = folder_name

    # Apply the limit_value if provided
    if limit_value:
        relation_file_folder = {}
        class_occurrences = defaultdict(list)

        for origin_data_path, folder_name in zip(file_paths, all_file_indices):
            limit_exceeded = False
            if folder_name in class_occurrences.keys() and len(class_occurrences[folder_name]) >= limit_value:
                limit_exceeded = True

            if not limit_exceeded:
                class_occurrences[folder_name].append(origin_data_path)

        for folder_name in class_occurrences.keys():
            for origin_data_path in class_occurrences[folder_name]:
                relation_file_folder[origin_data_path] = folder_name

    return relation_file_folder


def get_internal_asset(asset_name: str):
    with pkg_resources.path('base.assets', asset_name) as asset_path:
        return asset_path


def get_internal_asset_folder(asset_name: str):
    asset_path = pkg_resources.files('base.assets').joinpath(asset_name)
    return asset_path
