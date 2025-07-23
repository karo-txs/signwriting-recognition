#!/usr/bin/env python3
"""
Script: split_dataset.py
Divide um conjunto de imagens organizado em subpastas‑classe em pastas
`train/` e `test/`, preservando a proporção de amostras por classe.

Uso básico
----------
python split_dataset.py INPUT_DIR SPLIT_SIZE OUTPUT_DIR [--seed 42] [--move]

Posicionais
-----------
INPUT_DIR   Diretório que contém subpastas por classe (com as imagens).
SPLIT_SIZE  Quantidade de amostras **por classe** a colocar no split de teste.
OUTPUT_DIR  Diretório de saída, onde serão criadas as pastas train/ e test/

Opções
------
--seed N    Semente para reprodução da aleatoriedade (padrão = 42).
--move      Move, em vez de copiar, as imagens para a nova estrutura.
"""
import random
import shutil
import pathlib
import click
import csv


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "split_size",  # Quantidade de amostras que irá para treino.
    type=int,
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Semente para reprodução da aleatoriedade.",
)
@click.option(
    "--move",
    is_flag=True,
    default=False,
    help="Move em vez de copiar os arquivos.",
)
def split_dataset(
    input_dir: pathlib.Path,
    split_size: int,
    output_dir: pathlib.Path,
    seed: int,
    move: bool,
):
    """Implementação principal do script."""
    random.seed(seed)

    test_root = output_dir / "test"
    train_root = output_dir / "train"
    test_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    transfer = shutil.move if move else shutil.copy2

    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        images = sorted(class_dir.glob("*"))
        if not images:
            click.echo(f"[AVISO] Classe '{class_dir.name}' sem imagens — ignorando.")
            continue

        if split_size >= len(images):
            click.echo(
                f"[AVISO] split_size ({split_size}) ≥ nº imagens em '{class_dir.name}' "
                f"({len(images)}); todas irão para o conjunto de teste."
            )
            test_imgs, train_imgs = images, []
        else:
            train_imgs = random.sample(images, split_size)
            test_imgs = [img for img in images if img not in train_imgs]

        dst_test = test_root / class_dir.name
        dst_train = train_root / class_dir.name
        dst_test.mkdir(parents=True, exist_ok=True)
        dst_train.mkdir(parents=True, exist_ok=True)

        for img in test_imgs:
            transfer(img, dst_test / img.name)
        for img in train_imgs:
            transfer(img, dst_train / img.name)

        summary_rows.append(
            {
                "class": class_dir.name,
                "train": len(train_imgs),
                "test": len(test_imgs),
                "total": len(images),
            }
        )

        click.echo(
            f"[{class_dir.name}] train: {len(train_imgs)}  |  test: {len(test_imgs)}"
        )

    total_train = sum(r["train"] for r in summary_rows)
    total_test = sum(r["test"] for r in summary_rows)
    summary_rows.append(
        {
            "class": "TOTAL",
            "train": total_train,
            "test": total_test,
            "total": total_train + total_test,
        }
    )

    csv_path = output_dir / "split_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["class", "train", "test", "total"])
        writer.writeheader()
        writer.writerows(summary_rows)

    click.echo(f"\nResumo salvo em: {csv_path.resolve()}")
    click.echo("Processo concluído!")


if __name__ == "__main__":
    split_dataset()
