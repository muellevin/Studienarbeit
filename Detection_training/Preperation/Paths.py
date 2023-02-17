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
    LABELIMG_PATH : Final[str]

    def __init__(self) -> None:
        """Init paths"""
        self.IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, 'Tensorflow', 'workspace','images')
        self.COLLECTED_IMAGES_PATH = os.path.join(self.IMAGE_PATH, 'collected_images')
        self.RESIZED_IMAGES_PATH = self.COLLECTED_IMAGES_PATH + '_resized'
        self.TRAINSET_PATH = os.path.join(self.IMAGE_PATH, 'trainset')
        self.TESTSET_PATH = os.path.join(self.IMAGE_PATH, 'testset')
        self.DEVSET_PATH = os.path.join(self.IMAGE_PATH, 'devset')

        self.WORKSPACE_PATH = os.path.join(CURRENT_DIRECTORY, 'Tensorflow', 'workspace')
        self.SCRIPTS_PATH = os.path.join(CURRENT_DIRECTORY, 'Tensorflow','scripts')
        self.APIMODEL_PATH = os.path.join(CURRENT_DIRECTORY, 'Tensorflow','models')
        self.ANNOTATION_PATH = os.path.join(self.WORKSPACE_PATH,'annotations')
        self.MODEL_PATH = os.path.join(self.WORKSPACE_PATH, 'models')
        self.PRETRAINED_MODEL_PATH = os.path.join(self.WORKSPACE_PATH,'pre-trained-models')
        self.PROTOC_PATH =os.path.join(CURRENT_DIRECTORY, 'Tensorflow','protoc')
        self.LABELIMG_PATH = os.path.join(CURRENT_DIRECTORY, 'Tensorflow', 'labelimg')


    def setup_paths(self) -> None:
        """Setting up the paths"""

        paths = vars(self)
        for path in paths.values():
            print('Creating {}'.format(path))
            os.makedirs(path, exist_ok=True)


LABELS = [{'name':'raccoon', 'id':1}]
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

if __name__ == '__main__':
    print("hi")
    WorkingPaths().setup_paths()