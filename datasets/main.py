from interface.dataset import Dataset
import importlib
import inspect
import click
import os


def list_dataset_classes():
    dataset_classes = {}
    datasets_dir = "core"
    for filename in os.listdir(datasets_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = f"{datasets_dir}.{filename[:-3]}"
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Dataset) and obj is not Dataset:
                    dataset_classes[name.lower()] = obj
    return dataset_classes


@click.command()
@click.argument("datasets", nargs=-1)
def main(datasets):
    """
    CLI para baixar e mapear datasets. Exemplo de uso:

        python main.py DatasetA DatasetB
    """
    available_datasets = list_dataset_classes()

    if not datasets:
        click.echo("Datasets disponíveis:")
        for name in available_datasets:
            click.echo(f"- {name}")
        return

    for ds_name in datasets:
        ds_class = available_datasets.get(ds_name.lower())
        if not ds_class:
            click.echo(f"Dataset '{ds_name}' não encontrado.")
            continue

        click.echo(f"Processando dataset: {ds_name}")
        dataset_instance = ds_class()
        dataset_instance.download()
        dataset_instance.map_classes()


if __name__ == "__main__":
    main()
