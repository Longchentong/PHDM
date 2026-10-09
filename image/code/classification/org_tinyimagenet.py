import argparse
import shutil
from pathlib import Path


def organize(root):
    root = Path(root)
    annotations = root / 'val/val_annotations.txt'
    if not annotations.is_file() or not (root / 'train').is_dir():
        raise FileNotFoundError(f'Download Tiny-ImageNet before organizing {root}.')
    images = root / 'val/images'
    for line in annotations.read_text().splitlines():
        filename, label = line.split('\t')[:2]
        target = images / label
        target.mkdir(exist_ok=True)
        source = images / filename
        destination = target / filename
        if source.is_file():
            shutil.move(str(source), str(destination))
        elif not destination.is_file():
            raise FileNotFoundError(source)
    training_images = root / 'train/images'
    training_images.mkdir(exist_ok=True)
    for source in sorted((root / 'train').iterdir()):
        if source == training_images or not source.is_dir():
            continue
        destination = training_images / source.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(source), str(destination))
    print(root.resolve())


def main():
    parser = argparse.ArgumentParser(description='Organize Tiny-ImageNet for the PHDM image experiments.')
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'data/tiny-imagenet-200')
    organize(parser.parse_args().data_dir)


if __name__ == '__main__':
    main()
