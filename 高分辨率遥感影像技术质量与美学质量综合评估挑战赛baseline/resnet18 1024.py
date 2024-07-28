# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchmetrics.classification import Accuracy
from PIL import  Image
from barbar import Bar

class CustdomDataset(Dataset):
	def __init__(self, txtfile, img_dir, transforms=None):
		self.data = pd.read_csv(txtfile,delimiter=',', header=None, names=['image','label'],dtype={'imgage':str,'label':int})
		self.img_dir = img_dir
		self.transform = transforms
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		src_name = self.data.iloc[idx, 0]
		formatted_name = f'{int(src_name):04d}.jpg'
		img_name = os.path.join(self.img_dir, formatted_name)
		image = Image.open(img_name).convert('RGB')

		label = int(self.data.iloc[idx, 1])-1

		if self.transform:
			image = self.transform(image)
		return image, label



train_transforms = transforms.Compose([
	transforms.Resize(1024),
	transforms.CenterCrop(1024),
	transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.RandomPerspective(distortion_scale=0.1, p=0.3, interpolation=transforms.InterpolationMode.BILINEAR,
	                             fill=0),
	transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
	transforms.Resize(1024),
	transforms.CenterCrop(1024),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root_dir = '/media/gy/study/bisai/竞赛/评价/训练集/图片'

train_csv_file = 'train.txt'
val_csv_file = 'val.txt'

batch_size = 16
train_dataset = CustdomDataset(txtfile=train_csv_file, img_dir=root_dir, transforms=train_transforms)
val_dataset = CustdomDataset(txtfile=val_csv_file, img_dir=root_dir, transforms=val_transforms)


train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False, num_workers=8)



EPOCHS = 50
LR=0.0001
num_classes = 5
device = "cuda"
save_path = './resnet18_best.pth'

criterion = nn.CrossEntropyLoss().to(device)

model = timm.create_model('resnet18', pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


device = "cuda"
import time
best_acc = 0.0
for epoch in range(EPOCHS):
	train_loss = 0.00
	val_loss = 0.00

	print(f'Epoch {epoch + 1}')

	# Training loop
	time_start = time.time()
	model.train()
	train_acc_ = 0.0
	for idx, (inputs, labels) in enumerate(Bar(train_loader)):
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		predict_y = torch.max(outputs, dim=1)[1]

		loss = criterion(outputs, labels) 
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		train_acc_ += torch.eq(predict_y, labels.to(device)).sum().item()/(batch_size*1.0)

		# if idx%50==0:
		# 	print(f"Train Accuracy: {train_acc_/(idx+1)}")
	
	train_loss /= len(train_loader)
	train_acc_ /= len(train_loader)
	train_loss_formated = "{:.4f}".format(train_loss)
	print(f"Train Acc: {train_acc_}, 'Train loss: {train_loss}")
	# Validation loop

	train_time = time.time()
	# 打印花费时间长
	val_time = time.time()
	total_time = train_time - time_start
	epoch_mins = int(total_time / 60)
	epoch_secs = int(total_time % 60)
	print(f'Epoch {epoch} - Epoch Time(Train): {epoch_mins}m {epoch_secs}s')


	model.eval()
	val_acc_ = 0
	with torch.no_grad():
		for idx, (inputs, labels) in enumerate(Bar(val_loader)):#val_loader:
			# print(inputs.shape, labels.shape)
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			predict_y = torch.max(outputs, dim=1)[1]
			loss = criterion(outputs, labels) #+ loss_weight * center_loss(feat, labels)[0]
			val_loss += loss.item()

			val_acc_ += torch.eq(predict_y, labels.to(device)).sum().item()/(batch_size*1.0)


	val_acc_ /= len(val_loader)
	val_loss /= len(val_loader)
	val_loss_formated = "{:.4f}".format(val_loss)
	print(f"Val Accuracy: {val_acc_}, Validation Loss: {val_loss_formated}")

	#打印花费时间长
	val_time = time.time()
	total_time = val_time - train_time
	epoch_mins = int(total_time / 60)
	epoch_secs = int(total_time % 60)
	print(f'Epoch {epoch} - Epoch Time(Val): {epoch_mins}m {epoch_secs}s' )


	if val_acc_ > best_acc:
		print(f'Monitored acc has improved ({best_acc} --> {val_acc_}, starting save model')
		best_acc = val_acc_
		torch.save(model.state_dict(), save_path)
	torch.save(model.state_dict(), './resnet18_last.pth')
		
print('Finished Training,best_acc:',best_acc)



