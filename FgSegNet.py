#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np # linear algebra\
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX
import random
import time

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# In[2]:


image_dir = "../orderedImages"
gt_dir = "../orderedTruths"
image_name = list()
for path, _dir, file_name in os.walk(image_dir):
    image_name.extend(file_name)
image_name = [x for x in image_name if x.endswith(".jpg")]
image_name.sort()

gt_name = list()
for path, _dir, file_name in os.walk(gt_dir):
    gt_name.extend(file_name)
gt_name = [x for x in gt_name if x.endswith(".bmp")]
gt_name.sort()

# In[3]:


assert len(image_name) == len(gt_name)


# In[4]:


image_num = len(image_name)


# In[5]:


image_list = list()
gt_list = list()
for i in range(image_num):
    image_path = os.path.join(image_dir, image_name[i])
    gt_path = os.path.join(gt_dir, gt_name[i])
    
    tmp_list = list()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tmp_list.append(img)
    for j in range(2):
        img = cv2.pyrDown(img)
        tmp_list.append(img)
    
    image_list.append(tmp_list)
    gt = cv2.imread(gt_path)[:, :, 0]
    gt[gt >= 1] = 1
    if i == 0:
       for j in range(3):
            print(tmp_list[j].shape)
       print(gt.shape)
    gt_list.append(gt)


# In[6]:


"""plt.subplot(131)
plt.imshow(image_list[0][0])
plt.subplot(132)
plt.imshow(image_list[0][1])
plt.subplot(133)
plt.imshow(image_list[0][2])"""


# In[7]:


class FgSegNet_M(nn.Module):
    def __init__(self):
        super(FgSegNet_M, self).__init__()
        """triple CNN version"""
        """Encoder"""
        """Block 1"""
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_maxP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        """Block 2"""
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_maxP2 = nn.MaxPool2d(kernel_size=2, stride=2)
        """Block 3"""
        self.encoder_conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        """Block 4"""
        self.encoder_conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.encoder_dropout1 = nn.Dropout2d(p=0.5)
        self.encoder_conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.encoder_dropout2 = nn.Dropout2d(p=0.5)
        self.encoder_conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.encoder_dropout3 = nn.Dropout2d(p=0.5)
        
        """Decoder"""
        """Block 5"""
        self.decoder_conv1 = nn.ConvTranspose2d(512*3, 64, kernel_size=1, stride=1, padding=0)
        self.decoder_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(64, 512, kernel_size=1, stride=1, padding=0)
        """Block 6"""
        self.decoder_conv4 = nn.ConvTranspose2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.decoder_conv5 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2) # upsampling
        self.decoder_conv6 = nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, padding=0)
        """Block 7"""
        self.decoder_conv7 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.decoder_conv8 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.decoder_conv9 = nn.ConvTranspose2d(64, 128, kernel_size=1, stride=1, padding=0)
        """Block 8"""
        self.decoder_conv10 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2) # upsampling
        self.decoder_conv11 = nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1)
        
    def forward(self, img_list):
        assert len(img_list) == 3
        # assert img_list[0].shape[0] == self.img_height and img_list[0].shape[1] == self.img_height
        # assert img_list[1].shape[0] == img_list[0].shape[0] // 2 and             img_list[1].shape[1] == img_list[0].shape[1] // 2
        # assert img_list[2].shape[0] == img_list[0].shape[0] // 4 and             img_list[2].shape[1] == img_list[0].shape[1] // 4
        img_list = [torch.FloatTensor(x).unsqueeze(0).permute(0, 3, 1, 2) for x in img_list]
        if cuda:
           for i in range(3):
               img_list[i] = img_list[i].cuda()
        
        """Encoder"""
        encoder_out = list()
        # print(img_list[2].shape)
        for i in range(len(img_list)):
            out = F.relu(self.encoder_conv1(img_list[i]))
            out = F.relu(self.encoder_conv2(out))
            out = self.encoder_maxP1(out)
            
            out = F.relu(self.encoder_conv3(out))
            out = F.relu(self.encoder_conv4(out))
            out = self.encoder_maxP2(out)
            
            out = F.relu(self.encoder_conv5(out))
            out = F.relu(self.encoder_conv6(out))
            out = F.relu(self.encoder_conv7(out))
            
            out = F.relu(self.encoder_conv8(out))
            out = self.encoder_dropout1(out)
            out = F.relu(self.encoder_conv9(out))
            out = self.encoder_dropout2(out)
            out = F.relu(self.encoder_conv10(out))
            out = self.encoder_dropout3(out)
            encoder_out.append(out)
            
        """Decoder"""
        encoder_out[0] = F.interpolate(encoder_out[1], size=img_list[2].shape[2:], mode="bilinear")
        # print("Encoder out shape: ", encoder_out[0].shape)
        encoder_out[1] = F.interpolate(encoder_out[1], size=encoder_out[0].shape[2:], mode="bilinear")
        encoder_out[2] = F.interpolate(encoder_out[2], size=encoder_out[0].shape[2:], mode="bilinear")
        out = torch.cat(encoder_out, dim=1)
        out = F.relu(self.decoder_conv1(out))
        out = F.relu(self.decoder_conv2(out))
        out = F.relu(self.decoder_conv3(out))
        out = F.relu(self.decoder_conv4(out))
        # print("Before upsampling shape: ", out.shape)
        out = F.relu(self.decoder_conv5(out, output_size=(out.shape[0], out.shape[1], img_list[1].shape[2], img_list[1].shape[3])))
        out = F.relu(self.decoder_conv6(out))
        out = F.relu(self.decoder_conv7(out))
        out = F.relu(self.decoder_conv8(out))
        out = F.relu(self.decoder_conv9(out))
        out = F.relu(self.decoder_conv10(out, output_size=(out.shape[0], out.shape[1], img_list[0].shape[2], img_list[0].shape[3])))
        # print("Decoder 10 out shape: ", out.shape)
        out = F.sigmoid(self.decoder_conv11(out))
        
        return out


# In[8]:


net = FgSegNet_M()
if cuda:
    net.cuda()

train_index = list(np.random.choice(range(image_num), 800, replace=False))
test_index = [i for i in range(image_num) if i not in train_index]
random.shuffle(train_index)
random.shuffle(test_index)

optim = torch.optim.Adam(net.parameters(), lr=5e-2, weight_decay=1e-3, amsgrad=True)

def train(train_iter=1):
    lst_train_loss = 1e8
    lst_valid_loss = 1e8
    not_improve = 0
    edition = str(time.asctime(time.localtime(time.time())))
    for t in range(train_iter):
        print("Epoch {}".format(t))
        train_loss = 0.0
        for sample_id in train_index:
            predict_mask = net(image_list[sample_id])
            predict_mask = torch.squeeze(predict_mask, 0)
            predict_mask = torch.squeeze(predict_mask, 0)
            target_mask = torch.FloatTensor(gt_list[sample_id])
            if cuda:
                target_mask = target_mask.cuda()
            loss = F.binary_cross_entropy(predict_mask, target_mask)
            # print(loss)
            # print("Loss of image {} is {}".format(sample_id, loss))
            train_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        valid_loss = 0.0
        for sample_id in test_index:
            predict_mask = net(image_list[sample_id])
            predict_mask = predict_mask.detach()
            predict_mask = torch.squeeze(predict_mask, 0)
            predict_mask = torch.squeeze(predict_mask, 0)
            target_mask = torch.FloatTensor(gt_list[sample_id])
            if cuda:
                target_mask = target_mask.cuda()
            loss = F.binary_cross_entropy_with_logits(predict_mask, target_mask)
            valid_loss += loss.item()
        
        if train_loss < lst_train_loss and valid_loss < lst_valid_loss:
            torch.save(net.state_dict(), "./model/FgSegNet_" + edition)
            lst_train_loss = train_loss
            lst_valid_loss = valid_loss
        else:
            not_improve += 1
            if not_improve > 100:
                print("Early stop after {} epochs!".format(not_improve))
                break
        print("\tTrain loss: {}; Valid loss: {}".format(train_loss / len(train_index), valid_loss / len(test_index)))
# In[9]:

train(int(1e3))
