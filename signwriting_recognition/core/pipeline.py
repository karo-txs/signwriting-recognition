from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DataPipeline:
    original_path: str
    target_path: str
    steps: List[Dict[str, Any]]
    dataset_name: str = field(default=None)
    save_intermediate_steps: bool = field(default=False)
    last_intermediate_step_data: any = field(default=None)
    last_intermediate_step_path: str = field(default=None)

    def __post_init__(self):
        if not self.dataset_name:
            self.dataset_name = self._generate_dataset_name()

    def _generate_dataset_name(self) -> str:

        norm_str = None
        for step in self.steps:
            if step.get("name") == "normalization":
                norm_str = "norm"
                break

        for step in self.steps:
            if step.get("name") == "landmark-detector":
                detector_str = step.get("model")
                if detector_str.lower() == "mediapipe":
                    detector_str = "mp"
                else:
                    detector_str = detector_str.lower()
                break

        aug_str = None
        for step in self.steps:
            if step.get("name") == "augmentation":
                factor = step.get("factor")
                methods = step.get("methods", [])
                # Ex: "aug5[rotate,perturb]"
                aug_str = f"aug{factor}-{'_'.join(methods)}"
                break

        sampler_str = None
        for step in self.steps:
            if step.get("name") == "sampler":
                factor = step.get("factor")
                sampler_str = f"sampler{factor}"
                break

        # Exemplo de output: "mp_norm_sampler25_aug5[rotate_finger,perturb_points]"
        parts = [detector_str]
        if norm_str:
            parts.append(norm_str)
        if sampler_str:
            parts.append(sampler_str)
        if aug_str:
            parts.append(aug_str)

        dataset_name = "_".join(parts)
        return dataset_name


@dataclass
class TrainPipeline:
    train_dataset_paths: List[Dict[str, Any]]
    val_dataset_paths: List[Dict[str, Any]]
    test_dataset_paths: List[Dict[str, Any]]
    label_names: str
    experiment_path: str
    models_params: List[Dict[str, Any]]

    train_dataset: Any = field(default=None)
    val_dataset: Any = field(default=None)
    test_dataset: Any = field(default=None)
    len_unique_classes: int = field(default=None)
    models: List[Any] = field(default=None)


@dataclass
class EvaluationPipeline:
    eval_path: str
    label_names: str
    models_path: List[Dict[str, Any]]
    test_dataset_paths: List[Dict[str, Any]]
    
    actual_test_dataset: Any = field(default=None)
    actual_test_dataset_path: dict = field(default=None)
    actual_model: Any = field(default=None)
    actual_model_path: dict = field(default=None)
    len_unique_classes: int = field(default=None)


@dataclass
class Config:
    landmark_detector: str
    data_pipelines: List[DataPipeline] = field(default_factory=list)
    train_pipeline: List[TrainPipeline] = field(default_factory=list)
    evaluation_pipeline: List[EvaluationPipeline] = field(default_factory=list)
