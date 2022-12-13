# SCC-Tumor-Detection

## Notes
CNN-GNN model to identify and localize SCC tumors in WSIs. Basically the ArcticAI tumor detection model, 
but for SCC instead of BCC https://www.medrxiv.org/content/10.1101/2022.05.06.22274781v1.full-text

Data consists of 95 WSIs, each split into patches of 256x256 pixels. Here's some random patches from the dataset that I visualized.

![image](https://user-images.githubusercontent.com/96599771/207210035-313d5687-23c9-4b10-9671-dd006f662f56.png)

Here's the IDs of the 95 WSIs to make things simpler. (I didn't get rid of the ASAP_tumor_map part by the time that I realized I could've... oh well...)

IDs=['109_A1c_ASAP_tumor_map', '10_A1a_ASAP_tumor_map', '10_A1b_ASAP_tumor_map', '10_A2b_ASAP_tumor_map', '110_A2b_ASAP_tumor_map', '112_a_ASAP_tumor_map', '112_b_ASAP_tumor_map', '123_A1a_ASAP_tumor_map', '12_A1c_ASAP_tumor_map', '14_A1b_ASAP_tumor_map', '14_A2b_ASAP_tumor_map', '169_A2b_ASAP_tumor_map', '270_A1b_ASAP_tumor_map', '270_A1d_ASAP_tumor_map', '270_A1e_ASAP_tumor_map', '270_A2b_ASAP_tumor_map', '270_A2f_ASAP_tumor_map', '281_A1d_ASAP_tumor_map', '281_A1f_ASAP_tumor_map', '281_A2eX_ASAP_tumor_map', '311_A2c_ASAP_tumor_map', '327_A1a_ASAP_tumor_map', '327_A1d_ASAP_tumor_map', '327_B1c_ASAP_tumor_map', '341_a_ASAP_tumor_map', '341_b_ASAP_tumor_map', '342_a_ASAP_tumor_map', '342_b_ASAP_tumor_map', '343_a_ASAP_tumor_map', '343_b_ASAP_tumor_map', '343_c_ASAP_tumor_map', '343_d_ASAP_tumor_map', '344_a_ASAP_tumor_map', '344_b_ASAP_tumor_map', '345_a_ASAP_tumor_map', '345_b_ASAP_tumor_map', '346_a_ASAP_tumor_map', '346_b_ASAP_tumor_map', '350_A1a_ASAP_tumor_map', '350_A1b_ASAP_tumor_map', '350_A1c_ASAP_tumor_map', '350_A1d_ASAP_tumor_map', '350_A1e_ASAP_tumor_map', '351_A2b_ASAP_tumor_map', '352_A1d_ASAP_tumor_map', '352_A1e_ASAP_tumor_map', '352_A1g_ASAP_tumor_map', '352_A1h_ASAP_tumor_map', '352_A1i_ASAP_tumor_map', '353_A2b_ASAP_tumor_map', '354_A1b_ASAP_tumor_map', '354_A1c_ASAP_tumor_map', '354_A1d_ASAP_tumor_map', '354_A3a_ASAP_tumor_map', '354_A3b_ASAP_tumor_map', '354_A3c_ASAP_tumor_map', '354_D1b_ASAP_tumor_map', '355_A1d_ASAP_tumor_map', '356_A1b_ASAP_tumor_map', '358_A1a_ASAP_tumor_map', '358_A1b_ASAP_tumor_map', '361_a_ASAP_tumor_map', '361_b_ASAP_tumor_map', '362_A1a_ASAP_tumor_map', '362_A1b_ASAP_tumor_map', '362_A1c_ASAP_tumor_map', '363_A1b_ASAP_tumor_map', '363_A1c_ASAP_tumor_map', '363_A2b_ASAP_tumor_map', '363_A3b_ASAP_tumor_map', '364_A1b_ASAP_tumor_map', '364_A2b_ASAP_tumor_map', '364_A4b_ASAP_tumor_map', '365_A1b_ASAP_tumor_map', '365_A2b_ASAP_tumor_map', '366_A1a_ASAP_tumor_map', '366_A1b_ASAP_tumor_map', '366_A1c_ASAP_tumor_map', '367_A2b_ASAP_tumor_map', '368_A1b_ASAP_tumor_map', '368_A1c_ASAP_tumor_map', '368_A1d_ASAP_tumor_map', '369_A1b_ASAP_tumor_map', '369_A1c_ASAP_tumor_map', '369_A2b_ASAP_tumor_map', '370_A1b_ASAP_tumor_map', '370_A2a_ASAP_tumor_map', '370_A2b_ASAP_tumor_map', '37_A2d_ASAP_tumor_map', '61_A1a_ASAP_tumor_map', '61_B1a_ASAP_tumor_map', '70_A2b_ASAP_tumor_map', '7_A1c_ASAP_tumor_map', '7_A1d_ASAP_tumor_map', '7_A1e_ASAP_tumor_map']

### CNN
I trained the CNN using PathPreTrain https://github.com/jlevy44/PathPretrain/tree/master/pathpretrain

Here's the architecture

![Web capture_13-11-2022_221420_127 0 0 1](https://user-images.githubusercontent.com/96599771/207209812-69c0e268-ab6c-48c9-8c16-2de13dd159e8.jpeg)

I trained it for 10 epochs, batch size of 256, learning rate of 1e-3 for around 5.7 hours (not a long time, if I gave it some more time maybe the accuracy could improve). I used 80k patches for the training dataset and 10k for validation and testing. The AUC-ROC score and the accuracy could
definitely be improved using a different architecture (maybe EfficientNet? wanted to try that out but I haven't yet (https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html?adlt=strict&toWww=1&redig=424CA2642B0243BF84C2C8DB79B3A6FE) or freezing/unfreezing layers like you suggested. For now I'm just trying to finish the workflow and get a
decent accuracy and then going back to fine-tune everything and make adjustments, but if you could help me with that, it would be great :)

After training the CNN model, I generated embeddings using the model for each of the WSIs and saved the data in separate pickle files. I've used these embeddings
to generate a graph dataset to train a GCN to make connections between patches of WSIs to better localize and detect tumors in the WSIs.

![Web capture_24-11-2022_14521_127 0 0 1](https://user-images.githubusercontent.com/96599771/207213078-e5f49058-0f87-41eb-a9e9-00c4cb1d3dc1.jpeg) ![Web capture_24-11-2022_14521_127 0 0 10](https://user-images.githubusercontent.com/96599771/207213099-351bb191-de4d-4ab3-9178-ef71b1263eca.jpeg)

                                                       Embeddings

Data is located here:
```
/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Sophie_Chen/scc_tumor_data/prelim_patch_info_v2
/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Sophie_Chen/scc_tumor_data/prelim_patch_info 
```
v2 contains the pkl files with the annotations (i'm not sure what the annotations look like, i think they're just circles or rough outlines around areas with SCC, inflamm, etc.) ignore the pkl files in prelim_patch_info, they don't have annotations and they're the older version. 

prelim_patch_info contains the stacked npy arrays which are the actual WSIs split into patches.

Currently I'm working on constructing the GNN and training it, might need some help here too? I think Dr. Levy wants to finish the SCC ArcticAI studies soon
, so I'm going to start working on it more after my midterm exams are over (12/20, I've been procrastinating a bit, sorry). If you need anything else or have questions feel free to message me on slack! 
