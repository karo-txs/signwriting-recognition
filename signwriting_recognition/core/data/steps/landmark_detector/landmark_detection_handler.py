from core.data.steps.landmark_detector.mediapipe_landmark_detector import (
    MediapipeHandDetector,
)
from core.data.interfaces.hand_landmark_detector import HandLandmarkDetector
from core.utils.dict_functions import add_to_dict
from dataclasses import dataclass, field
from core.dtype import AbstractHandler
from core.pipeline import DataPipeline
from collections import defaultdict
from typing import Dict, List
from pathlib import Path
import logging

from core.utils import path_functions


@dataclass
class LandmarkDetectionHandler(AbstractHandler):
    """
    Executa detecção de marcos da mão no passo 'landmark-detector'.

    ▸ O detector é criado uma única vez (lazy-singleton).
    ▸ Suporte opcional a GPU (`use_gpu=True`) passado via step.
    """

    _detector: HandLandmarkDetector | None = field(init=False, default=None, repr=False)
    _model_name: str = field(init=False, default="mediapipe", repr=False)

    def _build_detector(self, use_gpu: bool = False) -> None:
        """Instancia o detector de forma preguiçosa."""
        if self._detector is None:
            self._detector = MediapipeHandDetector(use_gpu=use_gpu)

    def validate(self, request: DataPipeline) -> bool:
        step_cfg = next(
            (s for s in request.steps if s.get("name") == "landmark-detector"), None
        )
        if not step_cfg:
            return False

        self._model_name = step_cfg.get("model", "mediapipe")
        if self._model_name != "mediapipe":
            logging.error("Modelo '%s' não suportado.", self._model_name)
            return False

        self._build_detector(step_cfg.get("use_gpu", False))
        return True

    def handle(self, request: DataPipeline) -> DataPipeline:
        if not self.validate(request):
            return super().handle(request)

        logging.info(
            "DataPipeline: Run Hand Landmark Detection - model = %s", self._model_name
        )

        out_dir = Path(request.target_path) / "intermediate" / "2_landmark_detector"
        request.last_intermediate_step_path = str(out_dir)

        hand_labels: List[str] = []
        landmark_dict: Dict[str, List] = defaultdict(list)
        
        file_paths = path_functions.get_all_file_paths(request.original_path)
        label_names = path_functions.get_all_folder_names(request.original_path)

        relation_file_label = path_functions.get_relation_of_files_per_folder_name(
            file_paths, label_names, limit_value=None
        )
        request.last_intermediate_step_data = relation_file_label
        
        for img_path, label in request.last_intermediate_step_data.items():
            result = self._detector.detect_hand(
                image_path=img_path,
                label=label,
                save_dir=out_dir,
                save_landmark_image=request.save_intermediate_steps,
            )

            if not result:
                continue

            hand_labels.append(label)
            add_to_dict(landmark_dict, result)

        request.last_intermediate_step_data = {
            "hand_data_labels": hand_labels,
            "landmark_dict": dict(landmark_dict),
        }
        return super().handle(request)
