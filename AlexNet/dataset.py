import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_transforms(train = True):
    if train:
        return transforms.Compose([
            transforms.Resize(size = 256),
            transforms.RandomCrop(size = 224),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size = 256),
            transforms.RandomCrop(size = 224),
            transforms.ToTensor()
        ])

def get_dataset(root = "./CIFAR10/", train = True, transform = None, download = True):
    return dsets.CIFAR10(root = root,
                               train = train,
                               transform = transform,
                               download = download)

'''
In the original paper, the batch size was set to 128, 
but it was reduced to 8 in this implementation due to performance issues.
'''

def get_dataloader(dataset, batch_size = 8, shuffle = True): 
    return DataLoader(
        dataset = dataset, 
        batch_size = batch_size,
        shuffle = shuffle
    )
