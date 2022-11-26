#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().system('pip install -q ffmpeg-python')


# In[17]:


import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import ffmpeg
from IPython.display import Video
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import logging
from itertools import cycle

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


plt.style.use('ggplot')
cm = sns.light_palette("green", as_cmap=True)
pd.option_context('display.max_colwidth', 100)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# In[19]:


# SEED EVERYTHING
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)

# config
class config:
    BASE_DIR = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images"


# In[20]:


img_og = plt.imread("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_1/9101.jpg")
img_9101 = cv2.imread("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_1/9101.jpg")


# In[21]:


df = pd.read_csv("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train.csv")
train_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images"
df['image_path'] = train_dir + "/video_" + df['video_id'].astype(str) + "/" + df['video_frame'].astype(str) + ".jpg"
df.head().style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'})


# In[22]:


df.info() # lets check more details about the data


# In[23]:


df[df.annotations.str.len() > 2].head(5).style.background_gradient(cmap=cm) # filling up the annotation column


# In[24]:


df['annotations'] = df['annotations'].apply(eval)
df_train_v2 = df[df.annotations.str.len() > 0 ].reset_index(drop=True)
df_train_v2.head(5).style.background_gradient(cmap='Reds')


# # What is Sequence and its properties:

# In[25]:


df_train_v2["no_of_bbox"] = df_train_v2["annotations"].apply(lambda x: len(x))
df_train_v2["sequence"].value_counts(), len(df_train_v2["sequence"].value_counts())


# In[26]:


for i in range(3):
    print(df_train_v2["sequence"][df_train_v2["video_id"] == i].unique(), 
          df_train_v2["sequence"][df_train_v2["video_id"] == i].nunique())


# # Bounding box analysis in each video:

# In[27]:


def plot_with_count(df,vid):
    names = df["bbox_typ"].to_list()
    values = df["counts"].to_list()

    N = len(names)
    menMeans = values
    ind = np.arange(N)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(figsize=(15,6))

    ax.bar(ind,menMeans,width=0.4)
    plt.xticks(np.arange(0, N, step=1))
    plt.title(f"Number of bounding box VS Count of Bounding Box: Video{vid} ",fontsize=20)

    plt.xlabel('Number of bounding box', fontsize=18)
    plt.ylabel('Count', fontsize=16)

    for index,data in enumerate(menMeans):
        plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=15))


# In[28]:


vid = 0
df_vod0_bbox_cnt = df_train_v2["no_of_bbox"][df_train_v2["video_id"] == vid].value_counts().reset_index() # LEARNING .to_frame() and .reset_index()
df_vod0_bbox_cnt.columns = ['bbox_typ', 'counts']
plot_with_count(df_vod0_bbox_cnt,vid)


# In[29]:


vid = 1
df_vod1_bbox_cnt = df_train_v2["no_of_bbox"][df_train_v2["video_id"] == vid].value_counts().reset_index() # LEARNING .to_frame() and .reset_index()
df_vod1_bbox_cnt.columns = ['bbox_typ', 'counts']
plot_with_count(df_vod1_bbox_cnt,vid)


# In[30]:


vid = 2
df_vod2_bbox_cnt = df_train_v2["no_of_bbox"][df_train_v2["video_id"] == vid].value_counts().reset_index() # LEARNING .to_frame() and .reset_index()
df_vod2_bbox_cnt.columns = ['bbox_typ', 'counts']
plot_with_count(df_vod2_bbox_cnt,vid)


# In[33]:


# https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences
df = pd.read_csv("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train.csv")
df['annotations'] = df['annotations'].apply(eval)
df['n_annotations'] = df['annotations'].str.len()
df['has_annotations'] = df['annotations'].str.len() > 0
df['has_2_or_more_annotations'] = df['annotations'].str.len() >= 2
df['doesnt_have_annotations'] = df['annotations'].str.len() == 0
df['image_path'] = config.BASE_DIR + "video_" + df['video_id'].astype(str) + "/" + df['video_frame'].astype(str) + ".jpg"


# In[27]:


df_agg = df.groupby(["video_id", 'sequence']).agg({'sequence_frame': 'count', 'has_annotations': 'sum', 'doesnt_have_annotations': 'sum'})           .rename(columns={'sequence_frame': 'Total Frames', 'has_annotations': 'Frames with at least 1 object', 'doesnt_have_annotations': "Frames with no object"})
df_agg


# ### Future work:
# - Add plots on distribution of height and width of the bbox provided
# - Add plots on distribution of area of the bbox provided

# # 🎨 **Some Data Preparation:**

# In[34]:


def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(14, 14))
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance

dest_path1 = "./clahe_img"
os.mkdir(dest_path1)

for img_path in tqdm(df_train_v2["image_path"][0:400]):

    image = plt.imread(img_path)
    image_cv = cv2.imread(img_path)
    img_clahe = RecoverCLAHE(image_cv)
    file_name = img_path.split("/")[-1]
    
    cv2.imwrite(dest_path1+"/"+file_name, img_clahe)


# In[29]:


dest_path1 = "./annot_img"
os.mkdir(dest_path1)

idx = 0
for img_idx in tqdm(df_train_v2["image_path"][0:400]):
    file_name = img_idx.split("/")[-1] 
    img_path = os.path.join("./clahe_img",file_name)
    image = plt.imread(img_path)


    for i in range(len(df_train_v2["annotations"][idx])):
        file_name = img_path.split('/')[-1]
        b_boxs = df_train_v2["annotations"][idx][i]
        x,y,w,h = b_boxs["x"],b_boxs["y"],b_boxs["width"],b_boxs["height"]

        image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
        image = cv2.putText(image, 'starfish', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imwrite(dest_path1+"/"+file_name, image)
    idx +=1


# In[35]:


plt.figure(figsize = (12,15))


plt.rcParams["figure.figsize"] = [20.00, 10.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(figsize=(20,6))

ax.imshow(plt.imread("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0/1068.jpg"));
newax = fig.add_axes([0.3,0.3,0.6,0.7], anchor='NE', zorder=1)


newax.axis('off')
plt.show();


# ### Lets check we have images with same size or not:

# In[38]:


img_sizes = []
for i in df_train_v2["image_path"]:
    img_sizes.append(plt.imread(i).shape)

np.unique(img_sizes)


# In[41]:


# lets check total number of images with annotations
len(df_train_v2)


# In[39]:


plt.figure(figsize = (12,15))


plt.figure(figsize = (12,15))


plt.rcParams["figure.figsize"] = [20.00, 10.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(figsize=(20,6))

ax.imshow(plt.imread("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/results/annot_img/40.jpg"))

newax = fig.add_axes([0.26,0.2,0.6,0.6], anchor='NE', zorder=1)


newax.axis('off')
plt.show()


# In[40]:


def he_hsv(img_demo):
    img_hsv = cv2.cvtColor(img_demo, cv2.COLOR_RGB2HSV)

    # Histogram equalisation on the V-channel
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # convert image back from HSV to RGB
    image_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return image_hsv


# In[41]:


def plot_img(img_dir,num_items,func,mode):
    img_list = random.sample(os.listdir(img_dir), num_items)

    for i in range(len(img_list)):
        full_path = img_dir + '/' + img_list[i]
        img_temp1 = plt.imread(full_path)
        img_temp_cv = cv2.imread(full_path)
        plt.figure(figsize=(20,15))
        plt.subplot(1,2,1)
        plt.imshow(img_temp1);
        plt.subplot(1,2,2)
        if mode == 'plt':
            plt.imshow(func(img_temp1));
        elif mode == 'cv2':
            plt.imshow(func(img_temp_cv));


# In[42]:


vid_0_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0"
num_items1 = 4
plot_img(vid_0_dir,num_items1,he_hsv,"plt")


# In[43]:


def RecoverHE(sceneRadiance):
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
    return sceneRadiance

vid_0_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0"
num_items1 = 4
plot_img(vid_0_dir,num_items1,RecoverHE,"cv2")


# In[44]:


def clahe_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    clahe = cv2.createCLAHE(clipLimit = 15.0, tileGridSize = (20,20))
    v = clahe.apply(v)

    hsv_img = np.dstack((h,s,v))

    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    
    return rgb


vid_0_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0"
num_items1 = 4
plot_img(vid_0_dir,num_items1,clahe_hsv,"cv2")


# In[45]:


def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(14, 14))
    for i in range(3):

        
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))


    return sceneRadiance

vid_0_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0"
num_items1 = 4
plot_img(vid_0_dir,num_items1,RecoverCLAHE,"cv2")


# > <font size="4" face="verdana">
#             From the image, it seems like all the rocks, plants, and other underwater objects are on the ground and sunlight falling upon them, not inside a sea.
#         </font>

# In[46]:


def gamma_enhancement(image,gamma):
    R = 255.0
    return (R * np.power(image.astype(np.uint32)/R, gamma)).astype(np.uint8)

plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.imshow(img_og);
plt.subplot(2,2,2)
plt.imshow(gamma_enhancement(img_9101,1/0.6))

plt.subplot(2,2,3)
plt.imshow(img_og);
plt.subplot(2,2,4)
plt.imshow(gamma_enhancement(img_og,1/0.6))


# In[47]:


def RecoverGC(sceneRadiance):
    sceneRadiance = sceneRadiance/255.0
    
    for i in range(3):
        sceneRadiance[:, :, i] =  np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 3.2)
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance

vid_0_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0"
num_items1 = 4
plot_img(vid_0_dir,num_items1,RecoverGC,"cv2")


# In[48]:


import numpy as np

def global_stretching(img_L,height, width):
    I_min = np.min(img_L)
    I_max = np.max(img_L)
    I_mean = np.mean(img_L)

    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            p_out = (img_L[i][j] - I_min) * ((1) / (I_max - I_min))
            array_Global_histogram_stretching_L[i][j] = p_out

    return array_Global_histogram_stretching_L

def stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        Max_channel  = np.max(img[:,:,k])
        Min_channel  = np.min(img[:,:,k])
        for i in range(height):
            for j in range(width):
                img[i,j,k] = (img[i,j,k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel)+ 0
    return img

from skimage.color import rgb2hsv,hsv2rgb
import numpy as np



def  HSVStretching(sceneRadiance):
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = global_stretching(s, height, width)

    img_v_stretching = global_stretching(v, height, width)

    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255

    

    return img_rgb

def sceneRadianceRGB(sceneRadiance):

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance


# In[53]:


def RecoverICM(img1):
    img = stretching(img1)
    sceneRadiance = sceneRadianceRGB(img)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    
    return sceneRadiance


vid_0_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0"
num_items1 = 4
plot_img(vid_0_dir,num_items1,RecoverICM,"cv2")


# In[50]:


def plot_img_tf(img_dir,num_items,func,mode):
    img_list = random.sample(os.listdir(img_dir), num_items)
    full_path = img_dir + '/' + img_list[0]
    img_temp_plt = plt.imread(full_path)
    img_temp_cv = cv2.imread(full_path)
    if mode=="plt":
        
        img_stack = np.hstack((img_temp_plt,func(img_temp_plt)))
        plt.figure(figsize=(20,15))
        plt.imshow(img_stack);
        plt.title("Original Image VS Enhanced Image",fontsize=25)
        plt.axis("off")
        plt.show()
    if mode=="cv2":
        
        img_stack = np.hstack((img_temp_cv,func(img_temp_cv)))
        plt.figure(figsize=(20,15))
        plt.imshow(img_stack);
        plt.title("Original Image VS Enhanced Image",fontsize=25)
        plt.axis("off")
        plt.show()
    
    
    for i in range(1, len(img_list)):
        full_path = img_dir + '/' + img_list[i]
        img_temp_plt = plt.imread(full_path)
        img_temp_cv = cv2.imread(full_path)
        if mode=="plt":
            img_stack = np.hstack((img_temp_plt,func(img_temp_plt)));
            plt.figure(figsize=(20,15))
            plt.imshow(img_stack);
            plt.axis("off")
            plt.show()
        if mode=="cv2":
            img_stack = np.hstack((img_temp_cv,func(img_temp_cv)));
            plt.figure(figsize=(20,15))
            plt.imshow(img_stack);
            plt.axis("off")
            plt.show()

img_dir = "C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/train_images/video_0"
num_items = 4
plot_img_tf(img_dir,num_items,tfa.image.equalize,"plt")


# # 📺 **Lets checkout How it looks like as a video:**

# In[54]:


Video("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/results/img_movie.mp4")


# In[55]:


Video("C:/Users/kolap/Downloads/tensorflow-great-barrier-reef/results/annot_movie.mp4")

