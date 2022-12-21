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
from torchvision import transforms


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


device


# # Relationship Between Patches in a WSI 
# - Need to figure out how to turn a WSI into a graph, and how to know the relationships between different patches
# - If the x,y coords represent the center of each patch, I suppose what we can say is that adjacent patches need to be sqrt(2x^2) away if they are truly neighboring patches. 
# - Apparently, two nodes might be neighbors if their embeddings are sufficiently similar. 
# - Josh says that if patches are sqrt(2)*256 away from each other, then we should consider them neighbors 

# In[4]:


df = pd.read_pickle("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Sophie_Chen/scc_tumor_data/prelim_patch_info_v2/109_A1c_ASAP_tumor_map.pkl")


# In[5]:


df


# In[6]:


#create node -> (x,y) map 
pos_map = {}


# In[7]:


for index, row in tqdm(df.iterrows()):
    pos_map[index] = (row["x"], row["y"])


# # Create edge adj. list
# - Here we will use a library because n^2 is too slow
# - I am relying on the standard indexing in the df. Meaning, the first row should be patch 1, the second should be patch 2, and so on... 

# In[8]:


import dgl
import torch


# In[9]:


nodes = torch.tensor([])


# In[10]:


for i in tqdm(pos_map):
    x = torch.tensor([[float(pos_map[i][0]), float(pos_map[i][1])]])
    nodes = torch.cat((nodes, x), 0)


# In[11]:


len(nodes)


# In[12]:


# create graph where points sqrt(2)*256 away from each other are considered neighbors 
r_g, dist = dgl.radius_graph(nodes, 256*(2**(1/2)), get_distances=True) 


# In[13]:


r_g.edges()


# In[14]:


adj_list = list(zip(list(r_g.edges()[0]), list(r_g.edges()[1])))  #this is the adj list 


# In[15]:


for i in tqdm(range(len(adj_list))): #converting everything from tensors to ints 
    adj_list[i] = [adj_list[i][0].item(), adj_list[i][1].item()]


# In[16]:


adj_list


# In[17]:


adj_list = torch.tensor(adj_list)


# In[18]:


adj_list.shape


# # Create a graph data object 
# - Need to use torch.geometric here
# - Also need to define the model class
# - Also need to get embeddings for each patch here 
# - Also need the y matrix
# - We need to save all graph data objects in a seperate directory, and get their file location and map them to the meta file
# - The idea is that we will save all of this to a df that will contain columns= ["sample_id", "file_loc"] 

# ## Model Class

# In[19]:


#define the model class 
model = torch.hub.load('pytorch/vision', 'resnet50')
model.fc = nn.Sequential(nn.Linear(2048, 100), 
                         nn.ReLU(), 
                         nn.Dropout(p=.5), 
                         nn.Linear(100,2))


# In[20]:


model


# In[21]:


#load the best model 
model_path = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/Saved_Models/resnet50.pt"
model.load_state_dict(torch.load(model_path))


# In[22]:


#modify the model so that the output is the embedding layer 
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

model.fc = Identity() # remove the fc layer 
model.to(device)


# In[23]:


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ## Create Embeddings

# In[24]:


#get the patches
path = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Sophie_Chen/scc_tumor_data/prelim_patch_info/109_A1c_ASAP_tumor_map.npy"
arr = np.load(path)


# In[25]:


embed_dic = {} #map the patch_id -> embedding
patches = []


# In[26]:


for patch_id in tqdm(range(0, len(arr))):
    patch = preprocess(arr[patch_id])
    patches.append((patch_id, patch))


# In[27]:


patches_loader = DataLoader(dataset = patches, batch_size = 1)


# In[28]:


with torch.no_grad():
    for idx, patch in tqdm(patches_loader):
        patch = patch.to(device=device)
        embed = model(patch)
        embed_dic[idx.item()] = embed.detach().cpu().tolist()[0]


# In[29]:


len(embed_dic)
embeds = []


# In[30]:


for patch_idx in embed_dic:
    embeds.append(embed_dic[patch_idx])


# In[31]:


embeds = torch.tensor(embeds)


# In[32]:


embeds.shape


# In[33]:


adj_list = adj_list.T


# In[34]:


y = torch.tensor(list(df["scc"])) # this is the scc for each patch 


# In[35]:


y.shape


# In[36]:


from torch_geometric.data import Data


# In[37]:


data = Data(x=embeds, edge_index=adj_list, y=y)


# In[38]:


#see if you can save this object 
torch.save(data, "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/graph_data/109_A1c_ASAP.pt")


# In[39]:


#see if you can load it 
recovered_data = torch.load("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/graph_data/109_A1c_ASAP.pt")


# In[40]:


recovered_data


# In[41]:


#combine all of this into a function

#takes: df of WSI, np array of WSI, and model 
#returns: saves the graph data in a directory 
def create_graph(df, arr, model, preprocess, device, save_dir, sample_id):
    
    #create node -> (x,y) map 
    pos_map = {}
    for index, row in tqdm(df.iterrows()):
        pos_map[index] = (row["x"], row["y"])
        
    #use these nodes, which are made sequentially, to create a graph and eventually and edge list
    nodes = torch.tensor([])
    for i in tqdm(pos_map):
        x = torch.tensor([[float(pos_map[i][0]), float(pos_map[i][1])]])
        nodes = torch.cat((nodes, x), 0)
    
    # create graph where points sqrt(2)*256 away from each other are considered neighbors 
    r_g, dist = dgl.radius_graph(nodes, 256*(2**(1/2)), get_distances=True) 
    
    #get the adj_list
    adj_list = list(zip(list(r_g.edges()[0]), list(r_g.edges()[1])))  #this is the adj list 
    for i in tqdm(range(len(adj_list))): #converting everything from tensors to ints 
        adj_list[i] = [adj_list[i][0].item(), adj_list[i][1].item()]
        
    #make it a tensor
    adj_list = torch.tensor(adj_list).T
    
    #now, create embeddings for all of the patches within the WSI 
    embed_dic = {} #map the patch_id -> embedding
    patches = []
    
    for patch_id in tqdm(range(0, len(arr))): #get (idx, patch array)
        patch = preprocess(arr[patch_id])
        patches.append((patch_id, patch))

    patches_loader = DataLoader(dataset = patches, batch_size = 1)

    with torch.no_grad():
        for idx, patch in tqdm(patches_loader): # get the embeddings here 
            patch = patch.to(device=device)
            embed = model(patch)
            embed_dic[idx.item()] = embed.detach().cpu().tolist()[0]
    
    #now create an array for these embeddings 
    embeds = []
    for patch_idx in embed_dic:
        embeds.append(embed_dic[patch_idx])
    embeds = torch.tensor(embeds)
    
    #get the SCC array 
    y = torch.tensor(list(df["scc"])) # this is the scc for each patch 
    
    #make the graph data object 
    data = Data(x=embeds, edge_index=adj_list, y=y)
    
    #save this object 
    print(torch.save(data, save_dir+sample_id+".pt"))


# In[42]:


#test function 
save_dir = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/graph_data/"
sample_id = "109_A1c_ASAP"

create_graph(df = df, arr= arr, model = model, preprocess=preprocess, device=device, save_dir = save_dir, sample_id = sample_id)


# # Create & Save Graphs From All of the WSI 

# In[47]:


metadata = pd.DataFrame()
#here we will store the sample ids and their paths 
samples = []
paths = []


# In[48]:


#now create an np array containing all of the included patches from the relevant tumor maps
parent_dir = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Sophie_Chen/scc_tumor_data/prelim_patch_info/"
save_dir = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/graph_data/"

df_dir = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Sophie_Chen/scc_tumor_data/prelim_patch_info_v2/"


# In[ ]:


#iterate through all of the WSI samples
for f in tqdm(os.listdir(parent_dir)):
    id, ext = f.split(".")
    if ext == "npy":
        id = id[0:id.find("tumor")-1]
        print(id)

        #load the df for this sample 
        df = pd.read_pickle(df_dir + id + "_tumor_map.pkl")

        #load the np array for this sample 
        arr = np.load(parent_dir + id + "_tumor_map.npy")
        
        #create and save graph 
        create_graph(df = df, arr= arr, model = model, preprocess=preprocess, device=device, save_dir = save_dir, sample_id = id)
        
        #append sample_id and path 
        samples.append(id)
        paths.append(save_dir + id + ".pt")
        


# In[ ]:


metadata["sample_id"] = samples
metadata["path"] = paths


# In[ ]:


metadata.to_csv("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/graph_data/metadata.csv")

