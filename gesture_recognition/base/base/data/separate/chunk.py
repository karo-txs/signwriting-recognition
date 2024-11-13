

def separate_list_in_chunks(data_list: list, chunk_size: int):
    """
    The `chunk_generator` function takes a list of data and a chunk size as input, and yields chunks of
    the data list based on the specified chunk size.

    Args:
        data_list (list): The `data_list` parameter is a list of elements that you want to split into
    chunks.
        chunk_size (int): The `chunk_size` parameter specifies the size of each chunk that the
    `chunk_generator` function will yield. It determines how many elements from the `data_list` will be
    included in each chunk.
    """
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


def separate_dict_in_chunks(dict_values: dict, chunk_size: int):
    """
    The function separates a dictionary into chunks of a specified size and yields one chunk at a time when called.

    :param dict_values: A dictionary of key-value pairs that you want to separate into chunks
    :type dict_values: dict
    :param chunk_size: The `chunk_size` parameter specifies the size of each chunk into which the dictionary
    will be divided
    :type chunk_size: int
    """
    items = list(dict_values.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i + chunk_size])
