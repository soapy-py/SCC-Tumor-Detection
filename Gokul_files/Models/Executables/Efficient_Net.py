#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import torch 
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import glob, os, pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
from matplotlib import cm


# In[22]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[23]:


device


# # Create Train/Test/Validation split 
# - The patches that fall into the train, val, and test sets need to be from entirely distinct patient samples/WSI samples

# In[24]:


#write a different data loader class 
class Patch_Class():
    def __init__(self, csv_path, root_dir, samples, transform=None):
        self.samples = samples # this will contain the WSI samples that we want to include in the dataset
        
        self.patch_frame = pd.read_csv(csv_path) #get the metadata 
        #adjust the metadata so that it only contains data from the samples we want
        self.patch_frame = self.patch_frame[self.patch_frame["ID"].isin(self.samples)]
        
        self.root_dir = root_dir
        self.transform = transform
        
        #we also need to build the patch dictionary, which maps sample_id to patch_id to status 
        self.patch_dict = {}
        self.build_dictionary()
        
        #here, we also need to load in all of the distinct np arrays for each directory
        self.data_dict = {}
        self.build_data()
        
    def build_data(self):
        #go through each sub dir in the main dir 
        for s_dir in tqdm(os.listdir(self.root_dir)):
            #again, only build data for the relevant samples
            if s_dir != "metadata.csv" and s_dir in self.samples:
                data = np.load(self.root_dir + s_dir +"/data.npy")
                self.data_dict[s_dir] = data #map the sample_id to the npy data 
                
    def build_dictionary(self):
        for sample in self.samples:
            #now, for each sample, make the dictionary
            self.patch_dict[sample] = {}
        for id, group in tqdm(self.patch_frame.groupby("ID")):
            #only build dic for the samples that are needed
            if id in self.samples:
                for idx, group2 in group.groupby("patch_index"):
                    self.patch_dict[id][idx] = (group2["scc"] == True)
            
    def __len__(self):
        return len(self.patch_frame)

    def __getitem__(self, index):        
        #1 is the file id
        sample_id = self.patch_frame.iloc[index, 1]
        patch_id = self.patch_frame.iloc[index, 8]
        #get the image as a numpy array 
        img = self.data_dict[sample_id][patch_id]
        
        #turn the array into a PIL image, so that it can be resized (this is done for the Efficient Net)
#         img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        #get y_label and one hot encode it
#         ohe = [0, 0]
        y_label = int(list(self.patch_dict[sample_id][patch_id])[0])
#         ohe[y_label] = 1
        y_label = torch.tensor(y_label)

        if self.transform: 
            img = self.transform(img)
        return (img, y_label)


# In[25]:


#transform the function according to the pytorch docs
from torchvision import transforms

# img_size = 256
preprocess = transforms.Compose([
#     transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[26]:


# the directories we need

path = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/data/metadata.csv"

root_dir = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/data/"


# In[27]:


#get all of the sample names 
samples = []
for f in os.listdir(root_dir):
    if f != "metadata.csv":
        samples.append(f)

#split the sample names into train/test ~75/25
train, test = torch.utils.data.random_split(samples, [21, 9])

#further split train into train/validation
train, val = torch.utils.data.random_split(train, [18, 3])


# In[28]:


# get all of the different kinds of patches 

train_patches = Patch_Class(path, root_dir, samples=set(train), transform = preprocess)
val_patches = Patch_Class(path, root_dir, samples=set(val), transform = preprocess)
test_patches = Patch_Class(path, root_dir, samples=set(test), transform = preprocess)


# In[29]:


print(test_patches.__getitem__(231))

print(len(test_patches))


# In[30]:


print(len(val_patches), len(test_patches))


# # Create the Dataloader
# - also subset the datasets because they're big

# In[31]:


#trim all datasets untill they are 1/10th of the size 

train_dataset, discard = torch.utils.data.random_split(train_patches, [int(len(train_patches)*.3), int(len(train_patches)*.7)+1])
print(len(train_dataset))

val_dataset, discard = torch.utils.data.random_split(val_patches, [int(len(val_patches)*.10), int(len(val_patches)*.9)+1])
print(len(val_dataset))

test_dataset, discard = torch.utils.data.random_split(test_patches, [int(len(test_patches)*.3), int(len(test_patches)*.7)+1])
print(len(test_dataset))


# In[32]:


batch_size = 32

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True)


# # Load Model
# - Also change the architecture slightly 

# In[33]:


torch.hub.list("pytorch/vision")


# In[34]:


model = torch.hub.load('pytorch/vision', 'efficientnet_v2_l', pretrained=True)


# In[35]:


#visualize the layers 
ct = 0
for child in model.children():
    print("Layer: %d" %(ct))
    print(child)
    ct += 1


# In[36]:


# #we can also set the first, say, n layers to be frozen, and leave the remaining layers unfrozen, as follows 
# thresh = 1
# ct = 0
# #here we freeze up to and including the 6th layer
# for child in model.children():
#     if ct <= thresh:
#         for param in child.parameters():
#             param.requires_grad = False
#         print(child, ct)
#         ct += 1


# In[37]:


#change the model architecture a bit (for vision transformer)
model.fc = nn.Sequential(nn.Linear(1000, 100), 
                         nn.ReLU(), 
                         nn.Dropout(p=.5), 
                         nn.Linear(100,2))
model

model.train()
model.to(device)


# # Model Training 
# - Still need to implement some standard data augmentation (i.e., rotation, flip, contrast, etc...)

# In[38]:


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    model.eval() #put model in testing
    num_correct = 0
    num_samples = 0
    correct = {0:0, 1:0}
    total = {0:0, 1:0}
    with torch.no_grad():
        for x, y, name in tqdm(loader):
            #put batches on gpu 
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            for i,j in zip(predictions, y):
                if i.item() == j.item():
                    correct[i.item()] +=1
                total[j.item()] += 1
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

        print(
              f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
          )
        acc = num_correct/num_samples
        #find the accuracies for each class 
        return acc, correct, total

    model.train()


# In[39]:


#hyperparams
learning_rate = 5e-4
num_epochs =10 #20 works well - it seems as tho it is a local min 


# Some notes
# 1. Might need to figure out another loss that works better with one hot encoding 
# 2. Also might need to figure out how to calc AUC-ROC 

# In[40]:


# Loss and optimizer
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.


criterion = nn.CrossEntropyLoss()
# criterion = tgm.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean') #experimenting with focal loss 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=5, verbose=True)

#arrays to track the training loss and validation loss 
training_loss = []
validation_acc = []

# Train Network
for epoch in range(num_epochs):
    losses = []
    num_correct = 0
    num_samples = 0
    #train part 
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        # print("Batch: %d. Loss: %f" %(batch_idx, loss))

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    mean_loss = sum(losses)/len(losses)
    training_loss.append(mean_loss)
    scheduler.step(mean_loss)

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")
    
    #model in test mode 
    model.eval()
    with torch.no_grad():
        acc = 0
        for x, y in tqdm(val_loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            #find the test loss
            loss = criterion(scores, y)


            #find the test accuracy 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        #calc total acc here 
        acc = (num_correct/num_samples).item()
        print(acc)
        validation_acc.append(acc)
    #put the model back in train mode
    model.train()


# # Find/Calc/and Make AUC-ROC plot

# In[45]:


from sklearn.metrics import roc_auc_score

softmax = nn.Softmax(dim=1)


# In[46]:


model.eval()

probabilities = torch.Tensor([])
ground_truth = torch.Tensor([])

with torch.no_grad():
    for x, y in tqdm(test_loader):
        x = x.to(device=device)
        y = y.to(device=device)
        #find the probs
        scores = softmax(model(x))
        
        #move to cpu
        scores = scores.detach().to("cpu")
        y = y.detach().to("cpu")
        
        #concat them 
        probabilities = torch.cat((probabilities, scores))
        ground_truth = torch.cat((ground_truth, y))
  


# In[47]:


#predict the whole test cohort AUC-ROC

roc_auc_score(ground_truth, probabilities[:, 1])


# In[ ]:


#from sophie's code - viz. the curve 
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# fpr and tpr of all thresohlds
true = ground_truth
preds = probabilities[:, 1]
fpr, tpr, threshold = metrics.roc_curve(true, preds)

#get the metrics 
roc_auc = metrics.auc(fpr, tpr)

#plot
plt.title('Test Cohort-wide AUC-ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[23]:


get_ipython().system('nvidia-smi')


# In[ ]:


torch.cuda.empty_cache()

