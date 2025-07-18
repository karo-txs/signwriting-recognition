from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HandLandmarkDetector(ABC):
    use_gpu: bool = False
    
    @abstractmethod
    def detect_hand(
        image_path: str | Path,
        label: str,
        save_dir: str | Path,
        save_landmark_image: bool = True,
    ):
        pass
