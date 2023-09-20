import os
import torch
import data_setup, model_builder, engine, utils

from torchvision import transforms

def main():
	epochs = 10
	batch_size = 32
	hidden_units = 10
	learning_rate = 0.001
	
	train_dir = "../data/pizza_steak_sushi/train"
	test_dir = "../data/pizza_steak_sushi/test"
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	data_transform = transforms.Compose([transforms.Resize(size = (64, 64)), transforms.ToTensor()])
	
	train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(train_dir, test_dir, data_transform, batch_size)
	
	model = model_builder.TinyVGG(input_features = 3, output_features = len(class_names), hidden_units = hidden_units).to(device)
	
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
	
	engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device)
	
	utils.save_model(model, "models", "05_going_modular_script_mode_tinyvgg_model.pth")

if __name__ == '__main__':
	main()