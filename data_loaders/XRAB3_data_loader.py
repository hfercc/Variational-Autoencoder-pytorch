import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data.Dataset import ImageDataset

class XRAB3_DataLoader:
    def __init__(self, args):
        if args.dataset == 'XRAB3':
            # Data Loading
            DATA_DIR = "/home/user1_team6/project/data/reduced_data_tensors/"
            kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

            train_indices = np.array([i for i in range(8000)])
            val_indices = np.array([i for i in range(8000, 10000)])
            test_indices = np.array([i for i in range(10000, 12551)])

            train_set = ImageDataset(DATA_DIR, train_indices)
            val_set = ImageDataset(DATA_DIR, val_indices)
            test_set = ImageDataset(DATA_DIR, test_indices)  

            self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
            self.val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
            
            self.classes = (0, 1)
            # TODO: Finish implementing this class
            """
            transform_train = transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                                           **kwargs)

            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                          **kwargs)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            """
            
        else:
            raise ValueError('The dataset should be XRAB3')
