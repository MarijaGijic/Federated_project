from pathlib import Path
from abc import ABC, abstractmethod

class BaseConverter(ABC):
    
    def __init__(self, raw_path, client_output_path, dataset_name):
        self.raw_path = Path(raw_path)
        self.client_output = Path(client_output_path)
        self.dataset_name = dataset_name

        self.images_dir = self.client_output / "images"
        self.annotations_file = self.client_output / "annotations.csv"

        self.images_dir.mkdir(parents=True, exist_ok=True)
   
    @abstractmethod
    def convert(self):
        pass