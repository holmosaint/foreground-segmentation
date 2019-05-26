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

parser = argparse.ArgumentParser(description="Foreground Segmentation for 2019 PKU Image Processing Course")
parser.add_argument("--gpu", default=[False], help="Whether to use gpu", nargs=1, type=bool)
parser.add_argument("--encoder", default=["resnet"], help="Choose the architecture of the encoder network: vgg, resnet or segnet", nargs=1)
parser.add_argument("--train_image_dir", default=["./orderedImages/train"], nargs=1, type=str)
parser.add_argument("--train_mask_dir", default=["./orderedTruths/train"], nargs=1, type=str)
parser.add_argument("--val_image_dir", default=["./orderedImages/val"], nargs=1, type=str)
parser.add_argument("--val_mask_dir", default=["./orderedTruths/val"], nargs=1, type=str)
parser.add_argument("--test_image_dir", default=["./orderedImages/test"], nargs=1, type=str)
parser.add_argument("--test_mask_dir", default=["./orderedTruths/test"], nargs=1, type=str)
parser.add_argument("--train", default=[False], nargs=1, type=bool)
parser.add_argument("--test", default=[False], nargs=1, type=bool)
parser.add_argument("--epoch", default=[int(1e3)], nargs=1, type=int)
parser.add_argument("--step_per_epoch", default=[100], nargs=1, type=int)
parser.add_argument("--batch", default=[32], nargs=1, type=int)
parser.add_argument("--aug", default=[None], nargs=1, type=str)

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


def train_keras_aug(train_image_list, train_mask_list, val_image_list, val_mask_list, epoch, step_per_epoch, batch, net, device, encoder, steps=30):
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
    train_image_generator = train_datagen.flow_from_directory(train_image_list, class_mode=None, batch_size=32, seed=1)
    train_mask_generator = train_datagen.flow_from_directory(train_mask_list, class_mode=None, batch_size=32, seed=1)
    train_generator = zip(train_image_generator, train_mask_generator)

    val_datagen = ImageDataGenerator()
    val_image_generator = val_datagen.flow_from_directory(val_image_list, class_mode=None, batch_size=32, seed=2)
    val_mask_generator = val_datagen.flow_from_directory(val_mask_list, class_mode=None, batch_size=32, seed=2)
    val_generator = zip(val_image_generator, val_mask_generator)

    lst_train_loss = 1e8
    lst_val_loss = 1e8
    not_improve = 0
    adam = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-3, amsgrad=True)
    for t in range(epoch):
        print("Epoch {}".format(t))
        train_loss = 0.0
        s = 0
        for gen in train_generator:
            image, masks = gen
            # print(masks.shape)
            masks = masks[:, :, :, 0:1]
            masks[masks>=1] = 1
            image = torch.FloatTensor(image).permute(0, 3, 1, 2).to(device)

            pre_mask = net(image)

            if encoder == "segnet":
                inv_masks = 1 - masks
                masks = np.concatenate((inv_masks, masks), axis=-1)

            masks = torch.FloatTensor(masks).to(device).permute(0, 3, 1, 2)
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
        for i in range(3):
            for gen in val_generator:
                image, masks = gen
                masks[masks>=1] = 1
                masks = masks[:, :, :, 0:1]
                image = torch.FloatTensor(image).permute(0, 3, 1, 2).to(device)

                pre_mask = net(image).detach()

                if encoder == "segnet":
                    inv_masks = 1 - masks
                    masks = np.concatenate((inv_masks, masks), axis=-1)

                masks = torch.FloatTensor(masks).to(device).permute(0, 3, 1, 2)
                loss = F.binary_cross_entropy(pre_mask, masks)
                loss = loss.mean(0)
                val_loss += loss.item()
                break

        if train_loss < lst_train_loss and val_loss < lst_val_loss:
            torch.save(net.state_dict(), "./model/FgSegNet"+edition)
            lst_train_loss = train_loss
            lst_val_loss = val_loss
            not_improve = 0
        else:
            not_improve += 1
            if not_improve == 100:
                print("Early Stop at epoch {}".format(t))
                break
            elif not_improve % 10 == 0 and not_improve > 0:
                adjust_learning_rate(optim)
        print("\tTrain loss: {}; Valid loss: {}".format(train_loss / step_per_epoch, val_loss / 6))


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
        

def test(test_image_list, test_mask_list, net, device, model_name="./model/FgSegNetSun May 26 18:08:13 2019"):
    net.load_state_dict(torch.load(model_name))
    test_sample_num = len(test_image_list)
    for s in range(test_sample_num):
        images = test_image_list[s]
        masks = test_mask_list[s]
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
        test(test_image_list, test_mask_list, net, device)
