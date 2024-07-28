# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import timm
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from barbar import Bar
from torchmetrics.classification import Accuracy,ConfusionMatrix,CohenKappa
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import  Image

torch.set_printoptions(precision=4,threshold=float('inf'),linewidth=200)
# from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

class CustdomDataset(Dataset):
	def __init__(self, root_dir,transforms=None):
		self.data = os.listdir(root_dir)
		self.data.sort()
		self.root_dir = root_dir
		self.transform = transforms
		
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		img_name =  self.data[idx]
		img_path = os.path.join(self.root_dir, img_name)
		image = Image.open(img_path).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image, img_name

	def extract_label(self, filename):
		label = filename.split('/')[0]
		return self.class_dict[label]
val_transforms = transforms.Compose([
	transforms.Resize(1024),
	# transforms.CenterCrop(1024),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



import logging



root_dir = '/media/gy/study/bisai/竞赛/评价/初赛测试集/初赛测试集/图片'
test_dataset = CustdomDataset(root_dir=root_dir, transforms=val_transforms)
test_loader = DataLoader(test_dataset,batch_size=8, shuffle=False, num_workers=8)

num_classes = 5


device = "cuda"
# model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes).to(device)


# import torchvision.models as models
# model = models.resnet50(pretrained=False)
# num_feature = model.fc.in_features
# model.fc = torch.nn.Linear(num_feature, num_classes)
# model = model.to(device)


model = timm.create_model('maxvit_tiny_tf_512', pretrained=True)
model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes)
model.default_cfg['input_size'] = (3,1024,1024)
model = model.to(device)
# model_weight_path='/media/gy/study/bisai/竞赛/评价/baseline/resNet50.pth'  #0.9691
model_weight_path='maxvit-tiny_1024.pth' #train 0.975, val0.906  online 96.72891
model.load_state_dict(torch.load(model_weight_path))



model.eval()




all_predictions = []
all_paths = []
all_predict_probs = []



with torch.no_grad():
	for idx, (inputs,  paths) in enumerate(Bar(test_loader)):#val_loader:
		model.eval()
		inputs = inputs.to(device)
		outputs = model(inputs)
		pre_ = nn.functional.softmax(outputs, dim=1)



		max_probs, predict = torch.max(pre_,1)
		all_predictions.extend(predict.cpu().numpy()+1)
		all_predict_probs.extend(max_probs.cpu().numpy())
		all_paths.extend(paths)


predict_label = pd.DataFrame()
predict_label['id'] = [os.path.basename(all_paths[i])[:-4] for i in range(len(all_paths))]
predict_label['predict_y'] = [all_predictions[i] for i in range(len(all_paths))]


predict_label[['id','predict_y']].to_csv('submit0727maxivit-t.txt',index=None,header=None)
print("pandas fininshed")