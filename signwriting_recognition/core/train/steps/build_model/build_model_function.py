from core.train.steps.build_model.models import (
    FullyConnectedEmbedderModel,
    FullyConnectedModel,
)


def build_model(model_params: dict, len_unique_classes: int, models_path: str):
    model = None

    if model_params.get("model") == "fully_connected_embedder":
        model = FullyConnectedEmbedderModel(models_path)
    elif model_params.get("model") == "fully_connected":
        model = FullyConnectedModel(models_path)

    if model:
        model.build_from_dict(model_params, len_unique_classes)
    return model
