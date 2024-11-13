
def add_to_dict(target_dict, new_data):
    """
    The function `add_to_dict` takes a dictionary and new data, and appends the new data to the
    dictionary values if the key already exists, or creates a new key-value pair if the key is not
    present.

    Args:
        target_dict: The `target_dict` parameter is a dictionary that you want to add new data to.
        new_data: The `new_data` parameter in the `add_to_dict` function is a dictionary containing
    key-value pairs that you want to add to the `target_dict`. Each key in `new_data` will be added to
    `target_dict`, and if the key already exists in `target_dict`, the corresponding
    """
    for key, value in new_data.items():
        if key in target_dict:
            target_dict[key].append(value)
        else:
            target_dict[key] = [value]
