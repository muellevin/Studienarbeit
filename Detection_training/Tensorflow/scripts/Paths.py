"""
Ths module se up the directories and defines some general Names
"""
import os
from typing import Final
# we are in a subfolder and need the parent
CURRENT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))

def singleton(cls):
    """singleton's"""
    instances = {}

    def get_instance():
        """single instance"""
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class WorkingPaths:
    COLLECTED_IMAGES_PATH: Final[str]
    RESIZED_IMAGES_PATH: Final[str]

    TRAINSET_PATH : Final[str]
    TESTSET_PATH : Final[str]
    DEVSET_PATH : Final[str]
    WORKSPACE_PATH : Final[str]
    SCRIPTS_PATH : Final[str]
    APIMODEL_PATH : Final[str]
    ANNOTATION_PATH : Final[str]
    IMAGE_PATH : Final[str]
    MODEL_PATH : Final[str]
    PRETRAINED_MODEL_PATH : Final[str]
    PROTOC_PATH : Final[str]

    def __init__(self) -> None:
        """Init paths"""
        self.IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, 'workspace','images')
        self.COLLECTED_IMAGES_PATH = os.path.join(self.IMAGE_PATH, 'collected_images')
        self.RESIZED_IMAGES_PATH = self.COLLECTED_IMAGES_PATH + '_resized'
        self.TRAINSET_PATH = os.path.join(self.IMAGE_PATH, 'trainset')
        self.TESTSET_PATH = os.path.join(self.IMAGE_PATH, 'testset')
        self.DEVSET_PATH = os.path.join(self.IMAGE_PATH, 'devset')

        self.WORKSPACE_PATH = os.path.join(CURRENT_DIRECTORY, 'workspace')
        self.SCRIPTS_PATH = os.path.join(CURRENT_DIRECTORY, 'scripts')
        self.APIMODEL_PATH = os.path.join(CURRENT_DIRECTORY, 'models')
        self.ANNOTATION_PATH = os.path.join(self.WORKSPACE_PATH,'annotations')
        self.MODEL_PATH = os.path.join(self.WORKSPACE_PATH, 'models')
        self.PRETRAINED_MODEL_PATH = os.path.join(self.WORKSPACE_PATH,'pre-trained-models')
        self.PROTOC_PATH =os.path.join(CURRENT_DIRECTORY, 'protoc')



    def setup_paths(self) -> None:
        """Setting up the paths"""

        paths = vars(self)
        for path in paths.values():
            print('Creating {}'.format(path))
            os.makedirs(path, exist_ok=True)


LABELS = [{'name':'Raccoon', 'id':1}]
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
DATASET_NAME = 'dataset.tar.gz'
TRAINSET_NAME = 'trainset.record'
TESTSET_NAME = 'testset.record'

paths = WorkingPaths

DATASET = os.path.join(paths.IMAGE_PATH, DATASET_NAME)
TF_RECORD_SCRIPT = os.path.join(paths.SCRIPTS_PATH, TF_RECORD_SCRIPT_NAME)
TRAINSET_RECORD_PATH = os.path.join(paths.ANNOTATION_PATH, TRAINSET_NAME)
TESTSET_RECORD_PATH = os.path.join(paths.ANNOTATION_PATH, TESTSET_NAME)
LABELMAP = os.path.join(paths.ANNOTATION_PATH, LABEL_MAP_NAME)
XML_TO_CSV = os.path.join(paths.SCRIPTS_PATH, "xml_to_csv.py")
CSV_FILE_NAME = "detections.csv"
CSV_FILE = os.path.join(paths.COLLECTED_IMAGES_PATH, CSV_FILE_NAME)
CSV_CONV = os.path.join(paths.SCRIPTS_PATH, "csv_conv", "Cargo.toml")
CSV_RESIZE = os.path.join(paths.SCRIPTS_PATH, "resize_csv", "Cargo.toml")
CSV_FILE_RESIZED = os.path.join(paths.RESIZED_IMAGES_PATH, CSV_FILE_NAME)
OPEN_IMAGES = os.path.join(paths.IMAGE_PATH, "open-images-v7")
OPEN_IMAGES_TRAIN = os.path.join(OPEN_IMAGES, "train")
OPEN_IMAGES_TEST = os.path.join(OPEN_IMAGES, "test")
OPEN_IMAGES_VALIDATION = os.path.join(OPEN_IMAGES, "validation")
SPLIT_DATASET = os.path.join(paths.SCRIPTS_PATH, "split_dataset", "Cargo.toml")

if __name__ == '__main__':
    WorkingPaths.setup_paths()