import os
import Tensorflow.scripts.Paths as pp
# %pip install ultralytics
from ultralytics import YOLO

paths = pp.paths
paths.setup_paths()
CUSTOM_MODEL_NAME = 'raccoon_pre_yolov8n_1088_Bbest_ep100_aug_alb'
YOLO_WEIGHTS = os.path.join(paths.MODEL_PATH, CUSTOM_MODEL_NAME, "weights")
YOLO_BEST = os.path.join(YOLO_WEIGHTS, 'best_saved_model')

# make yaml
with open(pp.YOLO_CONFIG_PATH, "w") as config:
    config.write("train: ../trainset/images\n")
    config.write("test: ../testset/images\n")
    config.write("val: ../devset/images\n\n")
    config.write('nc: {}\n'.format(len(pp.LABELS) + 1))
    config.write("names:\n   0: index_error,\n")
    for i in range(0, len(pp.LABELS)):
        label = pp.LABELS[i]
        yaml_name = '   {}: {}'.format(label['id'], label['name'])
        if i < len(pp.LABELS) - 1:
            config.write(f'{yaml_name}, \n')
        else:
            config.write(f'{yaml_name}\n')

# Load a model from the ultralytics hub
# yaml -> scratch
# .pt pretrained
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model
results = model.train(
    data=pp.YOLO_CONFIG_PATH,
    epochs=100,
    imgsz=1088,
    batch=-1, # Use best option
    augment=True,# currently it does not care
    hsv_s=0.1,# during night saturation is low increasing would not be much better
    hsv_v=0.7, # make it darker (brightness)
    degrees=15,#image rotation
    perspective=0.001,# max perspective distortion
    shear=4,# shearing of objects in image
    mixup=0.1,# mixin images togethter (probability)
    name=CUSTOM_MODEL_NAME)  # train the model

#results = model.val()  # evaluate model performance on the validation set

# Load the trained model
#model = YOLO(os.path.join(YOLO_WEIGHTS, "best.pt"), task='detect')

# it automatically converts for anything you need
# Known Bug all int quantization .tflite files are not working (no Detections) same goes with the edgetpu
# to fix this you need to manually install this PR https://github.com/ultralytics/ultralytics/pull/1695
# model.export(format='edgetpu', imgsz=320, data=pp.YOLO_CONFIG_PATH)

# eval on edgetpu only locally possible
#TFLITE_MODEL = os.path.join(YOLO_BEST, 'best_full_integer_quant.tflite')
#tflite = YOLO(TFLITE_MODEL, task='detect')
#tflite.val(data=pp.YOLO_CONFIG_PATH, imgsz=320)