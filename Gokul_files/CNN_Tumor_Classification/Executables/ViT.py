#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Create Train/Test/Validation split 
# - The patches that fall into the train, val, and test sets need to be from entirely distinct patient samples/WSI samples

# In[3]:


# #write a different data loader class 
# class Patch_Class():
#     def __init__(self, csv_path, root_dir, samples, transform=None):
#         self.samples = samples # this will contain the WSI samples that we want to include in the dataset
        
#         self.patch_frame = pd.read_csv(csv_path) #get the metadata 
#         #adjust the metadata so that it only contains data from the samples we want
#         self.patch_frame = self.patch_frame[self.patch_frame["ID"].isin(self.samples)]
        
#         self.root_dir = root_dir
#         self.transform = transform
        
#         #we also need to build the patch dictionary, which maps sample_id to patch_id to status 
#         self.patch_dict = {}
#         self.build_dictionary()
        
#         #here, we also need to load in all of the distinct np arrays for each directory
#         self.data_dict = {}
#         self.build_data()
        
#     def build_data(self):
#         #go through each sub dir in the main dir 
#         for s_dir in tqdm(os.listdir(self.root_dir)):
#             #again, only build data for the relevant samples
#             if s_dir != "metadata.csv" and s_dir in self.samples:
#                 data = np.load(self.root_dir + s_dir +"/data.npy")
#                 self.data_dict[s_dir] = data #map the sample_id to the npy data 
                
#     def build_dictionary(self):
#         for sample in self.samples:
#             #now, for each sample, make the dictionary
#             self.patch_dict[sample] = {}
#         for id, group in tqdm(self.patch_frame.groupby("ID")):
#             #only build dic for the samples that are needed
#             if id in self.samples:
#                 for idx, group2 in group.groupby("patch_index"):
#                     self.patch_dict[id][idx] = (group2["scc"] == True)
            
#     def __len__(self):
#         return len(self.patch_frame)

#     def __getitem__(self, index):        
#         #1 is the file id
#         sample_id = self.patch_frame.iloc[index, 1]
#         patch_id = self.patch_frame.iloc[index, 8]
#         #get the image as a numpy array 
#         img = self.data_dict[sample_id][patch_id]
        
#         #turn the array into a PIL image, so that it can be resized (this is done for the ViT)
#         img = Image.fromarray(img.astype('uint8'), 'RGB')
        
#         #get y_label and one hot encode it
# #         ohe = [0, 0]
#         y_label = int(list(self.patch_dict[sample_id][patch_id])[0])
# #         ohe[y_label] = 1
#         y_label = torch.tensor(y_label)

#         if self.transform: 
#             img = self.transform(img)
#         return (img, y_label)


# In[4]:

import wandb
wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="ResNet-Tumor-Classification",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 3e-4,
    "architecture": "ViT",
    "dataset": "SCC",
    "epochs": 30,
    "batch_size": 256, 
    "num_workers": 0, 
    }
)

#transform the function according to the pytorch docs
from torchvision import transforms
#add some image transforms 
# img_size = 224

augmentations = transforms.RandomApply(torch.nn.ModuleList(
            [transforms.RandomRotation((0,315)),
            transforms.ColorJitter(brightness=.3, contrast=.3),
            transforms.RandomSolarize(.3),
            transforms.RandomInvert(), 
            transforms.RandomAdjustSharpness(2),
            ]), p=0.4)

preprocess_augmentation = transforms.Compose([
    #these are the random transforms I got from my other derm project
    augmentations, 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#it doesn't make sense to do this because the val/test sets also use preprocess. So we need a unique one for train. 
preprocess_normal = transforms.Compose([
#     transforms.Resize((img_size, img_size)),
    #these are the random transforms I got from my other derm project
#     augmentations, 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[5]:


# # the directories we need

# path = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/data/metadata.csv"

# root_dir = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/data/"


# In[6]:


# #get all of the sample names 
# samples = []
# for f in os.listdir(root_dir):
#     if f != "metadata.csv":
#         samples.append(f)

# #split the sample names into train/test ~75/25
# train, test = torch.utils.data.random_split(samples, [21, 9])

# #further split train into train/validation
# train, val = torch.utils.data.random_split(train, [18, 3])


# In[7]:


# # get all of the different kinds of patches 

# train_patches = Patch_Class(path, root_dir, samples=set(train), transform = preprocess)
# val_patches = Patch_Class(path, root_dir, samples=set(val), transform = preprocess)
# test_patches = Patch_Class(path, root_dir, samples=set(test), transform = preprocess)


# In[8]:


# print(test_patches.__getitem__(231))

# print(len(test_patches))


# # Sophie's Data Version

# In[9]:


s_df = pd.read_csv("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Sophie_Chen/file_info.csv")


# In[10]:


train_df = pd.DataFrame(columns=["ID", "x", "y", "patch_size", "annotations", "y_true", "inflamm", "scc", "patch_idx"])
test_df = pd.DataFrame(columns=["ID", "x", "y", "patch_size", "annotations", "y_true", "inflamm", "scc", "patch_idx"])
val_df = pd.DataFrame(columns=["ID", "x", "y", "patch_size", "annotations", "y_true", "inflamm", "scc", "patch_idx"])


# In[11]:


for idx, row in tqdm(s_df.iterrows()):
    if row["set"] == "train":
        WSI_df = pd.read_pickle(row["patch_info_loc"])
        patch_idx = [int(i) for i in range(len(WSI_df))]
        WSI_df["patch_idx"] = patch_idx
        train_df = train_df.append(WSI_df)
        
    elif row["set"] == "test":
        WSI_df = pd.read_pickle(row["patch_info_loc"])
        patch_idx = [int(i) for i in range(len(WSI_df))]
        WSI_df["patch_idx"] = patch_idx
        test_df = test_df.append(WSI_df)
        
    elif row["set"] == "val":
        WSI_df = pd.read_pickle(row["patch_info_loc"])
        patch_idx = [int(i) for i in range(len(WSI_df))]
        WSI_df["patch_idx"] = patch_idx
        val_df = val_df.append(WSI_df)


# In[12]:


#transform the function according to the pytorch docs
from torchvision import transforms
#add some image transforms 
img_size = 224

augmentations = transforms.RandomApply(torch.nn.ModuleList(
            [transforms.RandomRotation((0,315)),
            transforms.ColorJitter(brightness=.3, contrast=.3),
            transforms.RandomSolarize(.3),
            transforms.RandomInvert(), 
            transforms.RandomAdjustSharpness(2),
            ]), p=0.4)

preprocess_augmentation = transforms.Compose([
    #these are the random transforms I got from my other derm project
    augmentations, 
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#it doesn't make sense to do this because the val/test sets also use preprocess. So we need a unique one for train. 
preprocess_normal = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    #these are the random transforms I got from my other derm project
#     augmentations, 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[29]:


#write a different data loader class 
class Patch_Class():
    def __init__(self, set_type, slides_df, meta_df, transform=None):
        self.transform = transform
        self.set_type = set_type
        
        self.slides_df= slides_df  #get patch data
        self.meta_df = meta_df #metadata
#         print(slides_df)
        self.data_dic = {} #get the mapping between WSI id and np array
        
        self.build_dic()

    def build_dic(self):
        for idx, row in tqdm(self.meta_df.iterrows()):
            if row["set"] == self.set_type:
                self.data_dic[row["IDs"]] = np.load(row["npy_loc"])
            
    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, index):        
        #1 is the file id
        sample_id = self.slides_df.iloc[index, 0] 
        sample_id = sample_id.split("_", 2)
        sample_id = sample_id[0] + "_" + sample_id[1] #edit the sample_id to cut off the suffix  
        
        patch_id = self.slides_df.iloc[index, 8]
        
        #These labels are exclusive - map scc to 1, inflamm to 2, benign to 0
        inflamm_label = self.slides_df.iloc[index, 6] #inflamm
        scc_label = self.slides_df.iloc[index, 7] #scc 
        y_label = None
        if scc_label == 1: #if either both scc + inflamm, or just scc, say scc
            y_label = 1
        elif inflamm_label == 1: #else if inflamm, say inflamm
            y_label = 2
        else: #else benign
            y_label = 0 
        
        #get the image as a numpy array 
        img = self.data_dic[sample_id][patch_id]
        
        #turn the array into a PIL image, so that it can be resized and transformed
        img = Image.fromarray(img.astype('uint8'), 'RGB') #this here takes a lot of time, and it considerably slows training
        
        #get y_label and one hot encode it
#         ohe = [0, 0]
#         ohe[y_label] = 1
        y_label = torch.tensor(y_label)
        if self.transform: 
            img = self.transform(img)
        return (img, y_label)


# In[30]:


train_data = Patch_Class("train", train_df, s_df, transform=preprocess_normal)
val_data = Patch_Class("val", val_df, s_df, transform=preprocess_normal)
test_data = Patch_Class("test", test_df, s_df, transform=preprocess_normal)


# # Create the Dataloader
# - also subset the datasets because they're big

# In[31]:


# #trim all datasets untill they are 1/10th of the size 

train_dataset = train_data
print(len(train_dataset))

val_dataset, discard = torch.utils.data.random_split(val_data, [int(len(val_data)*.10), int(len(val_data)*.9)+1])
print(len(val_dataset))

test_dataset, discard = torch.utils.data.random_split(test_data, [int(len(test_data)*.10), int(len(test_data)*.9)+1])
print(len(test_dataset))


# In[ ]:


# train_dataset = train_data
# val_dataset = val_data 
# test_dataset = test_data


# # Load Model
# - Also change the architecture slightly 

# In[18]:


# torch.hub.list("pytorch/vision")


# In[32]:


model = torch.hub.load('pytorch/vision', 'vit_b_32', pretrained=True)


# In[33]:


#visualize the layers 
ct = 0
for child in model.children():
    print("Layer: %d" %(ct))
    print(child)
    ct += 1

#set the dropout
def set_dropout(model, p = 0.5):
    for child in model.children():
        if isinstance(child, torch.nn.Dropout):
            child.p = p
        if child.children():
            for child2 in child.children():
                if isinstance(child2, torch.nn.Dropout):
                    child2.p = p
                if child2.children():
                    for child3 in child2.children():
                        if isinstance(child3, torch.nn.Dropout):
                            child3.p = p
                        if child3.children():
                            for child4 in child3.children():
                                if isinstance(child4, torch.nn.Dropout):
                                    child4.p = p
                                if child4.children():
                                    for child5 in child4.children():
                                        if isinstance(child5, torch.nn.Dropout):
                                            child5.p = p
set_dropout(model, p = 0.5)
# In[34]:


# #we can also set the first, say, n layers to be frozen, and leave the remaining layers unfrozen, as follows 
# thresh = -1
# ct = 0
# #here we freeze up to and including the 6th layer
# for child in model.children():
#     if ct <= thresh:
#         for param in child.parameters():
#             param.requires_grad = False
#         print(child, ct)
#         ct += 1


# In[35]:


#change the model architecture a bit (for vision transformer)
model.head = nn.Sequential(nn.ReLU(), 
                           nn.Dropout(p=.5), 
                           nn.Linear(1000, 2))
model

model.train()
model.to(device)


# In[36]:


# !nvidia-smi


# # Model Training 
# - Still need to implement some standard data augmentation (i.e., rotation, flip, contrast, etc...)

# In[ ]:


# #set up w&b here 
# import wandb
# wandb.login()
# # Define sweep config
# # sweep_configuration = {
# #     'method': 'bayes',
# #     'name': 'sweep',
# #     'metric': {'goal': 'maximize', 'name': 'val_auc'},
# #     'parameters': 
# #     {
# #         'batch_size': {'values': [64, 128, 256]},
# #         'epochs': {'values': [5, 10, 15]},
# #         'lr': {'max': 0.01, 'min': 0.0001}
# #      }
# # }

# # sweep_id = wandb.sweep(sweep=sweep_configuration, project="ResNet-Tumor-Classification")
    
# # # start a new wandb run to track this script
import wandb
wandb.login()
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="ResNet-Tumor-Classification",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 3e-4,
#     "architecture": "ViT",
#     "dataset": "SCC",
#     "epochs": 30,
#     "batch_size": 256, 
#     "num_workers": 0, 
#     }
# )


# In[37]:


from sklearn.metrics import roc_auc_score

# code from: https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    #creating a set of all the unique classes using the actual class list
    unique_class = set([0,1,2])
    roc_auc_dict = {}

    for per_class in tqdm(unique_class):
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


# In[38]:


from sklearn.metrics import roc_auc_score

scaler = torch.cuda.amp.GradScaler()
softmax = nn.Softmax(dim=1)


# Some notes
# 1. Might need to figure out another loss that works better with one hot encoding 
# 2. Also might need to figure out how to calc AUC-ROC 

# In[43]:


# Loss and optimizer
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.

def train_model():
#     run = wandb.init(project='ResNet-Tumor-Classification')
    
    #set up the hyperparams
    learning_rate = 3e-4
    num_epochs = 30
    batch_size = 256 
    print("Learning Rate: %f; num_epochs: %d; batch_size: %d" %(learning_rate, num_epochs, batch_size))
    iterations = 1
    eval_iterations = 1000
    #split the dataset 
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 32, eta_min=1e-5, verbose=True)
    
    #arrays to track the training loss and validation loss 
    training_loss = []
    validation_auc= []

    # Train Network
    for epoch in range(num_epochs):
        losses = []
        num_correct = 0
        num_samples = 0
        best_auc = 0.76
        #train part 
        for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            with torch.cuda.amp.autocast():
                scores = model(data)
                loss = criterion(scores, targets)
                # print("Batch: %d. Loss: %f" %(batch_idx, loss))

            losses.append(loss.item()) # add loss
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # gradient descent or adam step
            scaler.step(optimizer)
            scaler.update()
            
            # validation performance
            if iterations % eval_iterations == 0: #let's evaluate the model every 1000 iterations
                model.eval()
                predictions = torch.Tensor([])
                ground_truth = torch.Tensor([])
                #here, we can collect the AUC for each class
                with torch.no_grad():
                    for x, y in tqdm(test_loader):
                        x = x.to(device=device)
                        y = y.to(device=device)
                        #find the probs
                        scores = softmax(model(x))
                        scores = torch.argmax(scores, dim=1) #transform into indices
                        #move to cpu
                        scores = scores.detach().cpu()
                        y = y.detach().cpu()
                        #concat them 
                        predictions = torch.cat((predictions, scores))
                        ground_truth = torch.cat((ground_truth, y))
                        
                    aucs = roc_auc_score_multiclass(ground_truth, predictions)
                    eval_iteration_loss = sum(losses)/len(losses)
                    print(aucs, eval_iteration_loss)
                    validation_auc.append(aucs)
                    wandb.log({'train loss': eval_iteration_loss, 'val_inflamm_auc':aucs[2], 'val_scc_auc':aucs[1], 'val_benign_auc':aucs[0], 'val_auc': (aucs[0] + aucs[1]+ aucs[2])/3,'epoch':epoch})
                    
#                     #saving criteria - has to be above 0.90 for SCC auc 
#                     if (aucs[0] + aucs[1]+ aucs[2])/3 > 0.70 and aucs[1] > 0.90 and (aucs[0] + aucs[1]+ aucs[2])/3 > best_auc:
#                         best_auc = (aucs[0] + aucs[1]+ aucs[2])/3
#                         #check the test set performance 
#                         model.eval()
#                         predictions = torch.Tensor([])
#                         ground_truth = torch.Tensor([])

#                         with torch.no_grad():
#                             for x, y in tqdm(test_loader):
#                                 x = x.to(device=device)
#                                 y = y.to(device=device)
#                                 #find the probs
#                                 scores = softmax(model(x))
#                                 scores = torch.argmax(scores, dim=1)
#                                 #move to cpu
#                                 scores = scores.detach().cpu()
#                                 y = y.detach().cpu()

#                                 #concat them 
#                                 predictions = torch.cat((predictions, scores))
#                                 ground_truth = torch.cat((ground_truth, y))
#                             print("Test set", roc_auc_score_multiclass(ground_truth, predictions))
#                         #save model 
#                         PATH = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/CNN_Tumor_Classification/saved_models/resnet50.pt"
#                         torch.save(model.state_dict(), PATH)
                        
            iterations += 1
            
        #put the model back in train mode
        model.train()
        
        mean_loss = sum(losses)/len(losses)
        training_loss.append(mean_loss)
        scheduler.step(mean_loss)
        print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")


# In[44]:


train_model()


# # Find/Calc/and Make AUC-ROC plot

# In[ ]:


from sklearn.metrics import roc_auc_score

softmax = nn.Softmax(dim=1)


# In[ ]:


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
        scores = scores.to("cpu")
        y = y.to("cpu")
        
        #concat them 
        probabilities = torch.cat((probabilities, scores))
        ground_truth = torch.cat((ground_truth, y))
  


# In[ ]:


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


# In[ ]:




