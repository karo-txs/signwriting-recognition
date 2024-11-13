import os


def create_directories_for_file(file_path):
    """
    The function creates directories for a given file path if they do not already exist.

    Args:
        file_path: The `file_path` parameter is a string representing the path to a file for which you
    want to create directories.
    """
    directory_path = os.path.dirname(file_path)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
