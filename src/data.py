import glob
import os
import random
import torch.utils.data
from PIL import Image
from torchvision import transforms


def load_dataset(img_dir, transform=None):
    dog_dir = os.path.join(img_dir, "dog")
    cat_dir = os.path.join(img_dir, "cat")
    imgs = []
    imgs.extend(glob.glob(os.path.join(dog_dir, "*.jpg")))  # 添加猫狗数据
    imgs.extend(glob.glob(os.path.join(cat_dir, "*.jpg")))
    random.shuffle(imgs)  # 打乱猫狗数据集
    return DataSet(imgs, transform)


class DataSet(torch.utils.data.Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    # 作为迭代器必须要有的
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0  # 狗的label设为1，猫的设为0
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_data(data_dir="./data/", input_size=224, batch_size=48, train_val_split=0.8):
    transforms1 = transforms.RandomHorizontalFlip()
    transforms2 = transforms.RandomRotation(15)
    transforms3 = transforms.ColorJitter(contrast=0.5, saturation=0.4, hue=0.2)
    transforms4 = transforms.RandomGrayscale(p=0.2)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomChoice([transforms1, transforms2, transforms3, transforms4]),
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1)),
            transforms.ToTensor(),
            transforms.Normalize([0.4864, 0.4533, 0.4154], [0.2625, 0.2558, 0.2586])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.4864, 0.4533, 0.4154], [0.2625, 0.2558, 0.2586])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.4864, 0.4533, 0.4154], [0.2625, 0.2558, 0.2586])
        ]),
    }

    train_val_dataset = load_dataset(os.path.join(data_dir, 'train'), data_transforms['train'])
    test_dataset = load_dataset(os.path.join(data_dir, 'test'), data_transforms['test'])

    # 计算划分训练集和验证集的索引
    train_val_size = len(train_val_dataset)
    train_size = int(train_val_size * train_val_split)
    val_size = train_val_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader
