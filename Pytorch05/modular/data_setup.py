import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers = os.cpu_count()

def create_dataloader(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int = num_workers):
	train_data = datasets.ImageFolder(train_dir, transform)
	test_data = datasets.ImageFolder(test_dir, transform)

	class_names = train_data.classes

	train_dataloader = DataLoader(train_data, batch_size = batch_size, num_workers = num_workers, shuffle = True, pin_memory = True)
	test_dataloader = DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, shuffle = False, pin_memory = True)

	return train_dataloader, test_dataloader, class_names


