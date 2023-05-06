import argparse
import csv
import os
import shutil
import threading

from collections import namedtuple

Detection = namedtuple('Detection', ['filename', 'width', 'height', 'class_', 'xmin', 'ymin', 'xmax', 'ymax'])

def parse_args():
    parser = argparse.ArgumentParser(description='Convert image detections to CSV format and split into train, test, and dev sets.')
    parser.add_argument('input_dir', type=str, help='Input directory containing image files')
    parser.add_argument('output_dir', type=str, help='Output directory for train, test, and dev sets')
    parser.add_argument('train_percent', type=float, default=0.85, help='Percentage of images to include in train set')
    parser.add_argument('test_percent', type=float, default=0.1, help='Percentage of images to include in test set')
    parser.add_argument('dev_percent', type=float, default=0.05, help='Percentage of images to include in dev set')
    return parser.parse_args()

def get_image_filenames(input_dir):
    return [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.endswith(('.csv', '.xml'))]

def get_splits(images, train_percent, test_percent, dev_percent):
    train_size = int(len(images) * train_percent)
    test_size = int(len(images) * test_percent)
    train_set, rest = images[:train_size], images[train_size:]
    test_set, dev_set = rest[:test_size], rest[test_size:]
    return set(train_set), set(test_set), set(dev_set)

def load_detections(input_dir):
    detections_csv = os.path.join(input_dir, 'detections.csv')
    with open(detections_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [Detection(*row) for row in reader]

def write_csv(input_dir, output_dir, files, records):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, 'detections.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        for record in records:
            if record.filename in files:
                shutil.copy2(os.path.join(input_dir, record.filename), os.path.join(output_dir, record.filename))
                writer.writerow(record)
    print(f'Wrote {len(files)} records in {output_csv}')

def write_sets(input_dir, output_dir, train_set, test_set, dev_set, records):
    threads = []
    threads.append(threading.Thread(target=write_csv, args=(input_dir, os.path.join(output_dir, 'trainset'), train_set, records)))
    threads.append(threading.Thread(target=write_csv, args=(input_dir, os.path.join(output_dir, 'testset'), test_set, records)))
    threads.append(threading.Thread(target=write_csv, args=(input_dir, os.path.join(output_dir, 'devset'), dev_set, records)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def split_set(input_dir, output_dir, train_percent, test_percent, dev_percent):
    images = get_image_filenames(input_dir)
    train_set, test_set, dev_set = get_splits(images, train_percent, test_percent, dev_percent)
    records = load_detections(input_dir)
    write_sets(input_dir, output_dir, train_set, test_set, dev_set, records)

if __name__ == '__main__':
    args = parse_args()
    split_set(args.input_dir, args.output_dir, args.train_percent, args.test_percent, args.dev_percent)
