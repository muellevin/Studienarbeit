import fiftyone as fo
import Tensorflow.scripts.Paths as Paths

# setting up the paths
paths = Paths.WorkingPaths
paths.setup_paths() # type: ignore


# Customize where zoo datasets are downloaded
fo.config.dataset_zoo_dir = paths.IMAGE_PATH

print("Raccoon")
# This will take a hell lot of time since the .csv file will be downloaded for all labels and images (train has 2.2 GB)
fo.zoo.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    classes=["Raccoon"],
    max_samples=100000,
)
print("Squirrel")
# This will take a hell lot of time since the .csv file will be downloaded for all labels and images (train has 2.2 GB)
fo.zoo.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    classes=["Squirrel"],
    max_samples=100000,
)
print("Fox")
# This will take a hell lot of time since the .csv file will be downloaded for all labels and images (train has 2.2 GB)
fo.zoo.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    classes=["Fox"],
    max_samples=100000,
)
print("Bird")
# This will take a hell lot of time since the .csv file will be downloaded for all labels and images (train has 2.2 GB)
fo.zoo.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    split="validation",
    classes=["Bird"],
    max_samples=500,
)
print("Cat")
fo.zoo.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    split="validation",
    classes=["Cat"],
    max_samples=500,
)
print("Person")
fo.zoo.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    split="validation",
    classes=["Person"],
    max_samples=500,
)
