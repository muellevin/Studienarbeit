"""
Ths module se up the directories and defines some general Names
"""
import os
from typing import Final
# we are in a subfolder and need the parent
CURRENT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))

CUSTOM_MODEL_NAME = 'tomatoDetection320x320'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

LABELS = []
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
DATASET_NAME = 'dataset.tar.gz'
TRAINSET_NAME = 'trainset.record'
TESTSET_NAME = 'testset.record'

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
    CHECKPOINT_PATH : Final[str]
    OUTPUT_PATH : Final[str]
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
        self.CHECKPOINT_PATH = os.path.join(self.MODEL_PATH,CUSTOM_MODEL_NAME)
        self.OUTPUT_PATH = os.path.join(self.MODEL_PATH,CUSTOM_MODEL_NAME, 'export')
        self.PROTOC_PATH =os.path.join(CURRENT_DIRECTORY, 'Tensorflow','protoc')
        self.LABELIMG_PATH = os.path.join(CURRENT_DIRECTORY, 'Tensorflow', 'labelimg')


    def setup_paths(self) -> None:
        """Setting up the paths"""

        paths = vars(self)
        for path in paths.values():
            print('Creating {}'.format(path))
            os.makedirs(path, exist_ok=True)

from typing import Final

if __name__ == '__main__':
    print("hi")
    WorkingPaths().setup_paths()