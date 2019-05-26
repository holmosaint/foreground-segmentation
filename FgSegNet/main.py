import argparse
import time
import numpy as np
from network import Encoder
from data_generator import DataGenerator
import torch
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser(description="Foreground Segmentation for 2019 PKU Image Processing Course")
parser.add_argument("--gpu", default=[False], help="Whether to use gpu", nargs=1, type=bool)
parser.add_argument("--encoder", default=["resnet"], help="Choose the architecture of the encoder network: vgg or res", nargs=1)
parser.add_argument("--train_image_dir", default="./orderedImages/train", nargs=1)
parser.add_argument("--train_mask_dir", default="./orderedTruths/train", nargs=1)
parser.add_argument("--val_image_dir", default="./orderedImages/val", nargs=1)
parser.add_argument("--val_mask_dir", default="./orderedTruths/val", nargs=1)
parser.add_argument("--test_image_dir", default="./orderedImages/test", nargs=1)
parser.add_argument("--test_mask_dir", default="./orderedTruths/test", nargs=1)
parser.add_argument("--train", default=[False], nargs=1, type=bool)
parser.add_argument("--test", default=[False], nargs=1, type=bool)
parser.add_argument("--epoch", default=[int(1e3)], nargs=1, type=int)
parser.add_argument("--step_per_epoch", default=[100], nargs=1, type=int)
parser.add_argument("--batch", default=[32], nargs=1, type=int)

edition = str(time.asctime(time.localtime(time.time())))

def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 5


def train(train_image_list, train_mask_list, val_image_list, val_mask_list, epoch, step_per_epoch, batch, net):
    lst_train_loss = 1e8
    lst_val_loss = 1e8
    not_improve = 0
    train_sample_num = train_image_list.shape[0]
    val_sample_num = val_image_list.shape[0]
    adam = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-3, amsgrad=True)
    for t in range(epoch):
        print("Epoch {}".format(t))
        train_loss = 0.0
        val_loss = 0.0
        """Train"""
        for s in range(step_per_epoch):
            sample = np.random.choice(train_sample_num, batch)
            images = train_image_list[sample:, ]
            masks = train_mask_list[sample:, ]
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
            images = val_image_list[sample:, ]
            masks = val_mask_list[sample:, ]
            pre_mask = net(images).detech()
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

        print("\tTrain loss {}\tVal loss {}".format(train_loss, val_loss))
 

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

    if args_dict["cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder = Encoder(structure=args_dict["encoder_net"], cuda=args_dict["cuda"])
    train_image_list, train_mask_list = DataGenerator(data_dir=args_dict["train_image_dir"], gt_dir=args_dict["train_mask_dir"], augmentation=True)
    val_image_list, val_mask_list = DataGenerator(data_dir=args_dict["val_image_dir"], gt_dir=args_dict["val_mask_dir"], augmentation=True)
    # test_image_list, test_mask_list = DataGenerator(data_dir=args_dict["test_image_dir"], gt_dir=args_dict["test_mask_dir"], augmentation=False)

    if args_dict["train"]:
        train(train_image_list, train_mask_list, val_image_list, val_mask_list, 
            epoch=args_dict["epoch"], step_per_epoch=args_dict["step_per_epoch"], net=encoder, batch=args_dict["batch"])
    
    if args_dict["test"]:
        pass
