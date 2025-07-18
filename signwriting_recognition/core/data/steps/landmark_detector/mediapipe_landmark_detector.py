from core.data.interfaces.hand_landmark_detector import HandLandmarkDetector
from core.utils.landmark_functions import draw_landmarks_on_image
from core.utils.path_functions import get_internal_asset
from mediapipe.tasks.python import vision as mp_vision
from core.utils.image_functions import load_rgb
from mediapipe.tasks import python as mp_tasks
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MediapipeHandDetector(HandLandmarkDetector):
    """
    Empacota a API de tarefas do Mediapipe num objeto reutilizável.

    ▸ A criação de `create_from_options` é custosa; por isso, o detector
      é inicializado uma única vez e mantido na instância.
    """

    num_hands: int = 2
    model_path: Path = field(
        default_factory=lambda: Path(get_internal_asset("hand_landmarker.task"))
    )

    def __post_init__(self) -> None:
        base = mp_tasks.BaseOptions(
            model_asset_path=str(self.model_path),
            delegate=(
                mp_tasks.BaseOptions.Delegate.GPU
                if self.use_gpu
                else mp_tasks.BaseOptions.Delegate.CPU
            ),
        )
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base, num_hands=self.num_hands
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(opts)

    def detect_hand(
        self,
        image_path: str | Path,
        label: str,
        save_dir: str | Path,
        save_landmark_image: bool = True,
    ) -> dict | None:
        """
        Analisa a imagem, devolvendo apenas a mão “mais alta” (menor Y
        do punho) — útil para bases de dados onde há sobreposição.

        Retorna None quando nenhuma mão é detectada.
        """
        image_path = Path(image_path)

        mp_image = load_rgb(image_path)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        idx = self._highest_hand_index(result)
        lm = result.hand_landmarks[idx]
        lm_world = result.hand_world_landmarks[idx]
        handedness = result.handedness[idx]

        if save_landmark_image:
            save_dir = Path(save_dir) / label
            save_dir.mkdir(parents=True, exist_ok=True)
            draw_landmarks_on_image([lm], save_dir / image_path.name)

        return {
            "hand_landmark": [[p.x, p.y, p.z] for p in lm],
            "handedness": [h.score for h in handedness],
            "handedness_name": [h.category_name for h in handedness],
            "world_hand": [[p.x, p.y, p.z] for p in lm_world],
        }

    @staticmethod
    def _highest_hand_index(result: mp_vision.HandLandmarkerResult) -> int:
        """Índice da mão cujo pulso (landmark 0) tem menor coordenada y."""
        return min(
            range(len(result.hand_landmarks)),
            key=lambda i: result.hand_landmarks[i][0].y,
        )
