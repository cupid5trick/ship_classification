from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor
from torch.utils.data import DataLoader, random_split
from PIL import Image

import pathlib



data_dir = pathlib.Path(__file__).parent.parent / 'data/raw'

preprocessing_compose = Compose([
    RandomHorizontalFlip(),
    ToTensor(),
])

dataset = ImageFolder(
    root = data_dir,
    loader = pil_loader,
    transform = preprocessing_compose,
    is_valid_file = lambda f: f.endswith('.png'),
)

def train_test_split(dataset, validate_ratio = 0.1, test_ratio = 0.1):
    train_size = len(dataset) * (1 - test_ratio - validate_ratio)
    validate_size = len(dataset) * validate_ratio
    test_size = len(dataset) * test_ratio
    if not validate_ratio > 0:
        return random_split(dataset, [int(train_size), int(test_size)])
    train_set, validate_set, test_set =  random_split(dataset, [int(train_size), int(validate_size), int(test_size)])

    return train_set, validate_set, test_set

train_set,  validate_set, test_set = train_test_split(dataset)

batch_size = 20

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8)