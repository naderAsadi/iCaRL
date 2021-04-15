from torchvision import transforms

from data.iCIFAR100 import iCIFAR100

def get_train_dataset(data_path, img_size=32):
    train_transform = transforms.Compose([#transforms.Resize(img_size),
                                                transforms.RandomCrop((32,32),padding=4),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ColorJitter(brightness=0.24705882352941178),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    return iCIFAR100(data_path, transform=train_transform, download=True)

def get_test_dataset(data_path, img_size=32):
    test_transform = transforms.Compose([#transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    return iCIFAR100(data_path, test_transform=test_transform, train=False, download=True)
