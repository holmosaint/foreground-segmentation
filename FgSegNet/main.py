import argparse
import time
import numpy as np
from network import Encoder
from data_generator import DataGenerator
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

parser = argparse.ArgumentParser(description="Foreground Segmentation for 2019 PKU Image Processing Course")
parser.add_argument("--gpu", default=[False], help="Whether to use gpu", nargs=1, type=bool)
parser.add_argument("--encoder", default=["resnet"], help="Choose the architecture of the encoder network: vgg, resnet or segnet", nargs=1)
parser.add_argument("--train_image_dir", default=["./orderedImages/train"], nargs=1, type=str)
parser.add_argument("--train_mask_dir", default=["./orderedTruths/train"], nargs=1, type=str)
parser.add_argument("--val_image_dir", default=["./orderedImages/val"], nargs=1, type=str)
parser.add_argument("--val_mask_dir", default=["./orderedTruths/val"], nargs=1, type=str)
parser.add_argument("--test_image_dir", default=["./test_dir/orderedImages/"], nargs=1, type=str)
parser.add_argument("--test_mask_dir", default=["./test_dir/orderedTruths/"], nargs=1, type=str)
parser.add_argument("--train", default=[False], nargs=1, type=bool)
parser.add_argument("--test", default=[False], nargs=1, type=bool)
parser.add_argument("--epoch", default=[int(1e3)], nargs=1, type=int)
parser.add_argument("--step_per_epoch", default=[100], nargs=1, type=int)
parser.add_argument("--batch", default=[32], nargs=1, type=int)
parser.add_argument("--aug", default=[None], nargs=1, type=str)
parser.add_argument("--test_mask", default=[False], nargs=1, type=bool)

edition = str(time.asctime(time.localtime(time.time())))

def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 5


def train_aug(train_image_list, train_mask_list, val_image_list, val_mask_list, epoch, step_per_epoch, batch, net, device, encoder):
    lst_train_loss = 1e8
    lst_val_loss = 1e8
    not_improve = 0
    train_sample_num = train_image_list.shape[0]
    val_sample_num = val_image_list.shape[0]
    # print(train_image_list.shape)
    adam = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-3, amsgrad=True)
    print("Training started!")
    for t in range(epoch):
        print("Epoch {}".format(t))
        train_image_list, train_mask_list = shuffle(train_image_list, train_mask_list, random_state=0)
        train_loss = 0.0
        val_loss = 0.0
        """Train"""
        for s in range(0, train_sample_num, batch):
            sample = np.arange(s, min(s + batch, train_sample_num)).astype(np.int)
            images = train_image_list[sample, :]
            masks = train_mask_list[sample, :]
            if encoder == "segnet":
                inv_masks = 1 - masks
                masks = np.concatenate((inv_masks, masks), axis=-1)
            images = torch.FloatTensor(images).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            masks = torch.FloatTensor(masks).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            pre_mask = net(images)
            loss = F.binary_cross_entropy(pre_mask, masks)
            loss = loss.mean(0)
            train_loss += loss.item()

            # update
            adam.zero_grad()
            loss.backward()
            adam.step()

        """Validation"""
        for s in range(0, val_sample_num, batch):
            sample = np.arange(s, min(s + batch, val_sample_num)).astype(np.int)
            images = val_image_list[sample, :]
            masks = val_mask_list[sample, :]
            images = val_image_list[s]
            masks = val_mask_list[s]
            
            if encoder == "segnet":
                inv_masks = 1 - masks
                masks = np.concatenate((inv_masks, masks), axis=-1)

            images = torch.FloatTensor(images).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            masks = torch.FloatTensor(masks).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            pre_mask = net(images).detach()
            loss = F.binary_cross_entropy(pre_mask, masks)
            loss = loss.mean(0)
            val_loss += loss.item()

        if train_loss < lst_train_loss and val_loss < lst_val_loss:
            not_improve = 0
            lst_train_loss = train_loss
            lst_val_loss = val_loss
            torch.save(net.state_dict(), "./model/FgSegNet"+edition)
        else:
            not_improve += 1
            if not_improve == 100:
                print("Early Stop at epoch {}".format(t))
                break
            elif not_improve % 10 == 0 and not_improve > 0:
                adjust_learning_rate(adam)

        print("\tTrain loss {}\tVal loss {}".format(train_loss / train_sample_num * batch, val_loss / val_sample_num * batch))


def train_no_aug(train_image_list, train_mask_list, val_image_list, val_mask_list, epoch, step_per_epoch, batch, net, device, encoder):
    lst_train_loss = 1e8
    lst_val_loss = 1e8
    not_improve = 0
    train_sample_num = len(train_image_list)
    val_sample_num = len(val_mask_list)
    adam = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-3, amsgrad=True)
    print("Training started!")
    for t in range(epoch):
        print("Epoch {}".format(t))
        train_image_list, train_mask_list = shuffle(train_image_list, train_mask_list, random_state=0)
        train_loss = 0.0
        val_loss = 0.0
        """Train"""
        for s in range(0, train_sample_num, batch):
            images = train_image_list[s]
            masks = train_mask_list[s]

            if encoder == "segnet":
                inv_masks = 1 - masks
                masks = np.concatenate((inv_masks, masks), axis=-1)

            images = torch.FloatTensor(images).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            masks = torch.FloatTensor(masks).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            pre_mask = net(images)
            loss = F.binary_cross_entropy(pre_mask, masks)
            loss = loss.mean(0)
            train_loss += loss.item()

            # update
            adam.zero_grad()
            loss.backward()
            adam.step()

        """Validation"""
        for s in range(0, val_sample_num, batch):
            images = val_image_list[s]
            masks = val_mask_list[s]

            if encoder == "segnet":
                inv_masks = 1 - masks
                masks = np.concatenate((inv_masks, masks), axis=-1)
                
            images = torch.FloatTensor(images).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            masks = torch.FloatTensor(masks).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            pre_mask = net(images).detach()
            loss = F.binary_cross_entropy(pre_mask, masks)
            loss = loss.mean(0)
            val_loss += loss.item()

        if train_loss < lst_train_loss and val_loss < lst_val_loss:
            not_improve = 0
            lst_train_loss = train_loss
            lst_val_loss = val_loss
            torch.save(net.state_dict(), "./model/FgSegNet"+edition)
        else:
            not_improve += 1
            if not_improve == 100:
                print("Early Stop at epoch {}".format(t))
                break
            elif not_improve % 10 == 0 and not_improve > 0:
                adjust_learning_rate(adam)

        print("\tTrain loss {}\tVal loss {}".format(train_loss / train_sample_num * batch, val_loss / val_sample_num * batch))


def train_keras_aug(train_image_list, train_mask_list, val_image_list, val_mask_list, epoch, step_per_epoch, batch, net, device, encoder, steps=3):
    print("Train image: ", train_image_list)
    print("Train mask: ", train_mask_list)
    print("Val image: ", val_image_list)
    print("Val mask: ", val_mask_list)
    train_datagen = ImageDataGenerator(rotation_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
    train_image_generator = train_datagen.flow_from_directory(train_image_list, class_mode=None, batch_size=16, seed=1)
    train_mask_generator = train_datagen.flow_from_directory(train_mask_list, class_mode=None, batch_size=16, seed=1)
    train_generator = zip(train_image_generator, train_mask_generator)

    val_datagen = ImageDataGenerator()
    val_image_generator = val_datagen.flow_from_directory(val_image_list, class_mode=None, batch_size=32, seed=2)
    val_mask_generator = val_datagen.flow_from_directory(val_mask_list, class_mode=None, batch_size=32, seed=2)
    val_generator = zip(val_image_generator, val_mask_generator)

    lst_train_loss = 1e8
    lst_val_loss = 1e8
    not_improve = 0
    adam = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5, weight_decay=1e-3, amsgrad=True)
    model_name = "./model/SegNet-w-aug"
    net.load_state_dict(torch.load(model_name))

    for t in range(epoch):
        print("Epoch {}".format(t))
        train_loss = 0.0
        s = 0
        for gen in train_generator:
            image, masks = gen
            # print(masks.shape)
            image = image[:, :, :, :]
            masks = masks[:, :, :, 0:1]
            masks[masks>=1] = 1
            """bbox = mask2bbox(masks)
            mask_img = np.zeros(image.shape)
            for box in bbox:
                mask_img[box[0]:box[2], box[1]:box[3], :] = 1
            image = np.multiply(mask_img, image)"""

            image = torch.FloatTensor(image).permute(0, 3, 1, 2).to(device)
            # image = torch.FloatTensor(image).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            pre_mask = net(image)

            if encoder in ["segnet", "vgg_skip"]:
                inv_masks = 1 - masks
                masks = np.concatenate((inv_masks, masks), axis=-1)

            masks = torch.FloatTensor(masks).to(device).permute(0, 3, 1, 2)
            # masks = torch.FloatTensor(masks).unsqueeze(0).to(device).permute(0, 3, 1, 2)
            loss = F.binary_cross_entropy(pre_mask, masks)
            loss = loss.mean(0)
            train_loss += loss.item()

            # update
            adam.zero_grad()
            loss.backward()
            adam.step()

            s += 1
            if s == steps:
                break

        val_loss = 0
        for i in range(30):
            for gen in val_generator:
                image, masks = gen
                image = image[:, :, :, :]
                masks = masks[:, :, :, 0:1]
                masks[masks>=1] = 1
                # masks = masks[:, :, :, 0:1]
                """bbox = mask2bbox(masks)
                mask_img = np.zeros(image.shape)
                for box in bbox:
                   mask_img[box[0]:box[2], box[1]:box[3], :] = 1
                image = np.multiply(mask_img, image)"""
                  
                image = torch.FloatTensor(image).permute(0, 3, 1, 2).to(device)  
                # image = torch.FloatTensor(image).unsqueeze(0).permute(0, 3, 1, 2).to(device)

                pre_mask = net(image).detach()

                if encoder in ["segnet", "vgg_skip"]:
                    inv_masks = 1 - masks
                    masks = np.concatenate((inv_masks, masks), axis=-1)

                masks = torch.FloatTensor(masks).to(device).permute(0, 3, 1, 2)
                # masks = torch.FloatTensor(masks).unsqueeze(0).to(device).permute(0, 3, 1, 2)
                loss = F.binary_cross_entropy(pre_mask, masks)
                loss = loss.mean(0)
                val_loss += loss.item()
                break

        if train_loss < lst_train_loss and val_loss < lst_val_loss:
            torch.save(net.state_dict(), "./model/"+encoder+edition)
            lst_train_loss = train_loss
            lst_val_loss = val_loss
            not_improve = 0
        else:
            not_improve += 1
            if not_improve == 100:
                print("Early Stop at epoch {}".format(t))
                break
            elif not_improve % 10 == 0 and not_improve > 0:
                adjust_learning_rate(adam)
        print("\tTrain loss: {}; Valid loss: {}".format(train_loss / steps, val_loss / 30))


def train_dispatch(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, epoch, step_per_epoch, batch, net, device, aug, encoder):
    if aug is None:        
        train_image_list, train_mask_list = DataGenerator(data_dir=train_image_dir, gt_dir=train_mask_dir, augmentation=False)
        val_image_list, val_mask_list = DataGenerator(data_dir=val_image_dir, gt_dir=val_mask_dir, augmentation=False)
        train_no_aug(train_image_list, train_mask_list, val_image_list, val_mask_list, 
            epoch=epoch, step_per_epoch=step_per_epoch, net=net, batch=batch, device=device, encoder=encoder)
    elif aug == "keras":
        train_keras_aug(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, epoch, step_per_epoch, batch, net, device, encoder=encoder)
    else:
        train_image_list, train_mask_list = DataGenerator(data_dir=train_image_dir, gt_dir=train_mask_dir, augmentation=True)
        val_image_list, val_mask_list = DataGenerator(data_dir=val_image_dir, gt_dir=val_mask_dir, augmentation=True)
        train_aug(train_image_list, train_mask_list, val_image_list, val_mask_list, 
            epoch=epoch, step_per_epoch=step_per_epoch, net=net, batch=batch, device=device, encoder=encoder)
        

def test(test_image_list, test_mask_list, net, device, encoder):
    # model_name = "./model/" + encoder + edition
    model_name = "./model/segnetSat Jun  1 17:50:33 2019"
    net.load_state_dict(torch.load(model_name))
    test_sample_num = len(test_image_list)
    for s in range(test_sample_num):
        print("Processing image {}".format(30001 + s), end="\r")
        images = test_image_list[s]
        masks = test_mask_list[s]
        images = torch.FloatTensor(images).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        masks = torch.FloatTensor(masks).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        pre_mask = net(images).detach()
        # print(pre_mask.shape)
        # break
        """pre_mask = torch.argmax(pre_mask, dim=1, keepdim=True)
        pre_mask = pre_mask.squeeze(0).squeeze(0)
        pre_mask = pre_mask.cpu().numpy()
        np.savetxt("./result/" + str(30000 + s + 1) + ".txt", pre_mask, fmt="%.3f")"""
        pre_mask = pre_mask.squeeze(0)
        pre_mask = pre_mask.cpu().numpy()
        with open("./result/" + str(30000 + s + 1) + ".txt", "w") as outfile:
            for slice2d in pre_mask:
                np.savetxt(outfile, slice2d)



def mask2bbox(mask):
    """给定一个mask，返回所有的bbox: [y, x, h, w]"""
    lbl = label(mask) 
    props = regionprops(lbl)
    bbox = [(prop.bbox[0], prop.bbox[1], prop.bbox[2], prop.bbox[3]) \
            for prop in props if prop.bbox[3] - prop.bbox[1] > 3 and prop.bbox[2] - prop.bbox[0] > 3]
    return bbox

def test_mask(test_image_list, test_mask_list, net, device, encoder):
    # model_name = "./model/" + encoder + edition
    model_name = "./model/SegNet-w-aug"
    net.load_state_dict(torch.load(model_name))
    test_sample_num = len(test_image_list)
    for s in range(test_sample_num):
        print("Processing image {}".format(30001 + s), end="\r")
        images = test_image_list[s]
        masks = test_mask_list[s]
        bbox = mask2bbox(masks)
        mask_img = np.zeros(images.shape)

        for box in bbox:
            mask_img[box[0]:box[2], box[1]:box[3], :] = 1
        images = np.multiply(mask_img, images)
        images = torch.FloatTensor(images).to(device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        masks = torch.FloatTensor(masks).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        pre_mask = net(images).detach()
        pre_mask = torch.argmax(pre_mask, dim=1, keepdim=True)
        pre_mask = pre_mask.squeeze(0).squeeze(0)
        pre_mask = pre_mask.cpu().numpy()
        np.savetxt("./result/" + str(30000 + s + 1) + ".txt", pre_mask, fmt="%.3f")

if __name__ == "__main__":
    args = parser.parse_args()

    args_dict = dict()
    args_dict["cuda"] = args.gpu[0]
    args_dict["encoder_net"] = args.encoder[0]
    args_dict["train_image_dir"] = args.train_image_dir[0]
    args_dict["train_mask_dir"] = args.train_mask_dir[0]
    args_dict["val_image_dir"] = args.val_image_dir[0]
    args_dict["val_mask_dir"] = args.val_mask_dir[0]
    args_dict["test_image_dir"] = args.test_image_dir[0]
    args_dict["test_mask_dir"] = args.test_mask_dir[0]
    args_dict["is_train"] = args.train[0]
    args_dict["is_test"] = args.test[0]
    args_dict["epoch"] = args.epoch[0]
    args_dict["step_per_epoch"] = args.step_per_epoch[0]
    args_dict["batch"] = args.batch[0]
    args_dict["aug"] = args.aug[0]
    args_dict["test_mask"] = args.test_mask[0]

    if args_dict["cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder = Encoder(structure=args_dict["encoder_net"], cuda=args_dict["cuda"])
    net = encoder.net.to(device)

    if args_dict["is_train"]:
        train_dispatch(train_image_dir=args_dict["train_image_dir"], train_mask_dir=args_dict["train_mask_dir"], 
            val_image_dir=args_dict["val_image_dir"], val_mask_dir=args_dict["val_mask_dir"], 
            epoch=args_dict["epoch"], step_per_epoch=args_dict["step_per_epoch"], net=net, batch=args_dict["batch"], 
            device=device, encoder=args_dict["encoder_net"], aug=args_dict["aug"])
    
    if args_dict["is_test"]:
        test_image_list, test_mask_list = DataGenerator(data_dir=args_dict["test_image_dir"], gt_dir=args_dict["test_mask_dir"], augmentation=False)
        test(test_image_list, test_mask_list, net, device, encoder=args_dict["encoder_net"])

    if args_dict["test_mask"]:
        test_image_list, test_mask_list = DataGenerator(data_dir=args_dict["test_image_dir"], gt_dir=args_dict["test_mask_dir"], augmentation=False)
        test_mask(test_image_list, test_mask_list, net, device, encoder=args_dict["encoder_net"])
