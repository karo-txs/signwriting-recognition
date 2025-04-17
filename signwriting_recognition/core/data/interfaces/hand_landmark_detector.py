from abc import ABC, abstractmethod


class HandLandmarkDetector(ABC):
    @abstractmethod
    def detect_from_chunks(detector_name: str, image_path, label, save_path: str):
        pass
