import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16 as vgg
from torchvision.models import resnet18 as resnet
import sys
import cv2
import numpy as np

class Encoder():
    def __init__(self, structure="vgg", cuda=False):
        if structure not in ["vgg", "resnet", "segnet", "vgg_skip"]:
            print("The architecture of the encoder should be either 'vgg', 'resnet', 'segnet' or 'vgg_skip', got {} instead.".format(structure))
            sys.exit(0)
        
        if structure == "vgg":
            self.net = vgg_net(cuda=cuda)
        elif structure == "resnet":
            self.net = res_net(cuda=cuda)
        elif structure == "vgg_skip":
            self.net = vgg_skip()
        else:
            self.net = SegNet()


class vgg_net(nn.Module):
    def __init__(self, cuda):
        super(vgg_net, self).__init__()
        self.vgg_model = vgg(pretrained=True)
        self.cuda = cuda
        """triple CNN version"""
        """Encoder"""
        """Block 1"""
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_conv1.weight.data.copy_(self.vgg_model.features[0].weight.data)
        self.encoder_conv1.bias.data.copy_(self.vgg_model.features[0].bias.data)
        self.encoder_conv1.weight.requires_grad = False
        self.encoder_conv1.bias.requires_grad = False
        self.encoder_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2.weight.data.copy_(self.vgg_model.features[2].weight.data)
        self.encoder_conv2.bias.data.copy_(self.vgg_model.features[2].bias.data)
        self.encoder_conv2.weight.requires_grad = False
        self.encoder_conv2.bias.requires_grad = False
        self.encoder_maxP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        """Block 2"""
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_conv3.weight.data.copy_(self.vgg_model.features[5].weight.data)
        self.encoder_conv3.bias.data.copy_(self.vgg_model.features[5].bias.data)
        self.encoder_conv3.weight.requires_grad = False
        self.encoder_conv3.bias.requires_grad = False
        self.encoder_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_conv4.weight.data.copy_(self.vgg_model.features[7].weight.data)
        self.encoder_conv4.bias.data.copy_(self.vgg_model.features[7].bias.data)
        self.encoder_conv4.weight.requires_grad = False
        self.encoder_conv4.bias.requires_grad = False
        self.encoder_maxP2 = nn.MaxPool2d(kernel_size=2, stride=2)
        """Block 3"""
        self.encoder_conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_conv5.weight.data.copy_(self.vgg_model.features[10].weight.data)
        self.encoder_conv5.bias.data.copy_(self.vgg_model.features[10].bias.data)
        self.encoder_conv5.weight.requires_grad = False
        self.encoder_conv5.bias.requires_grad = False
        self.encoder_conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_conv6.weight.data.copy_(self.vgg_model.features[12].weight.data)
        self.encoder_conv6.bias.data.copy_(self.vgg_model.features[12].bias.data)
        self.encoder_conv6.weight.requires_grad = False
        self.encoder_conv6.bias.requires_grad = False
        self.encoder_conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_conv7.weight.data.copy_(self.vgg_model.features[14].weight.data)
        self.encoder_conv7.bias.data.copy_(self.vgg_model.features[14].bias.data)
        self.encoder_conv7.weight.requires_grad = False
        self.encoder_conv7.bias.requires_grad = False
        """Block 4"""
        self.encoder_conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.encoder_conv8.weight.data.copy_(self.vgg_model.features[17].weight.data)
        self.encoder_conv8.bias.data.copy_(self.vgg_model.features[17].bias.data)
        # self.encoder_conv8.weight.requires_grad = False
        # self.encoder_conv8.bias.requires_grad = False
        self.encoder_dropout1 = nn.Dropout2d(p=0.8)
        self.encoder_conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.encoder_conv9.weight.data.copy_(self.vgg_model.features[19].weight.data)
        self.encoder_conv9.bias.data.copy_(self.vgg_model.features[19].bias.data)
        # self.encoder_conv1.weight.requires_grad = False
        # self.encoder_conv1.bias.requires_grad = False
        self.encoder_dropout2 = nn.Dropout2d(p=0.8)
        self.encoder_conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.encoder_conv10.weight.data.copy_(self.vgg_model.features[21].weight.data)
        self.encoder_conv10.bias.data.copy_(self.vgg_model.features[21].bias.data)
        # self.encoder_conv1.weight.requires_grad = False
        # self.encoder_conv1.bias.requires_grad = False
        self.encoder_dropout3 = nn.Dropout2d(p=0.8)

        """Decoder"""
        """Block 5"""
        self.decoder_conv1 = nn.ConvTranspose2d(512*3, 64, kernel_size=1, stride=1, padding=0)
        self.decoder_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(64, 512, kernel_size=1, stride=1, padding=0)
        """Block 6"""
        self.decoder_conv4 = nn.ConvTranspose2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.decoder_conv5 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1) # upsampling
        self.decoder_conv6 = nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, padding=0)
        """Block 7"""
        self.decoder_conv7 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.decoder_conv8 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=1)
        self.decoder_conv9 = nn.ConvTranspose2d(64, 128, kernel_size=1, stride=1, padding=0)
        """Block 8"""
        self.decoder_conv10 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1) # upsampling
        self.decoder_conv11 = nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1)

    def forward(self, img_list):
        assert len(img_list) == 3
        # print(img_list[0].shape)
        # print(img_list[1].shape)
        # print(img_list[2].shape)
        # assert img_list[0].shape[0] == self.img_height and img_list[0].shape[1] == self.img_height
        # assert img_list[1].shape[0] == img_list[0].shape[0] // 2 and img_list[1].shape[1] == img_list[0].shape[1] // 2
        # assert img_list[2].shape[0] == img_list[0].shape[0] // 4 and img_list[2].shape[1] == img_list[0].shape[1] // 4
        if len(img_list[0].shape) == 4:
            img_list = [torch.FloatTensor(x).permute(0, 3, 1, 2) / 256.0 for x in img_list]
        else:
            img_list = [torch.FloatTensor(x).unsqueeze(0).permute(0, 3, 1, 2) / 256.0 for x in img_list]
        if self.cuda:
           for i in range(3):
               img_list[i] = img_list[i].cuda()

        """Encoder"""
        encoder_out = list()
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

        encoder_out[0] = F.interpolate(encoder_out[0], size=img_list[2].shape[2:], mode="bilinear")
        encoder_out[1] = F.interpolate(encoder_out[1], size=encoder_out[0].shape[2:], mode="bilinear")
        encoder_out[2] = F.interpolate(encoder_out[2], size=encoder_out[0].shape[2:], mode="bilinear")
        out = torch.cat(encoder_out, dim=1)

        """Decoder"""
        out = F.relu(self.decoder_conv1(encoder_out))
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class res_net(nn.Module):
    def __init__(self, cuda):
        super(res_net, self).__init__()
        self.cuda = cuda
        self.net = resnet(pretrained=True)
        """Delete the avg pooling and fc layer"""
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        # self.net.requires_grad = False
        print(self.net)

        """Decoder"""
        """Block 5"""
        self.decoder_block0 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # upsampling
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        ])
        """Block 6"""
        self.decoder_block1 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # upsampling
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
        ])
        """Block 7"""
        self.decoder_block2 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # upsampling
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
        ])
        """Block 8"""
        self.decoder_block3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # upsampling
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
        ])
        """Block 9"""
        self.decoder_block4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # upsampling
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1),
        ])


    def forward(self, image):
        out = self.net(image)
        # print(out.shape)
        out = self.decoder_block0(out)
        # print(out.shape)
        out = self.decoder_block1(out)
        # print(out.shape)
        out = self.decoder_block2(out)
        # print(out.shape)
        out = self.decoder_block3(out)
        # print(out.shape)
        out = F.sigmoid(self.decoder_block4(out))
        # print(out.shape)

        return out
        

class SegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self, num_classes=2, n_init_features=3, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SegNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return F.softmax(self.classifier(feat), dim=1)


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            if n_blocks == 3:
                layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True),
                           nn.Dropout(drop_rate)]
            else:
                layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            if n_blocks == 3:
                layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True),
                           nn.Dropout(drop_rate)]
            else:
                layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)


class vgg_skip(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self, num_classes=2, n_init_features=3, drop_rate=0.5,
                 filter_config=[64, 128, 256, 512]):
        super(vgg_skip, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = [2, 2, 3, 3, 3]
        encoder_filter_config = [n_init_features,] + filter_config
        decoder_n_layers = [3, 3, 3, 2, 1]
        decoder_filter_config = filter_config[::-1] + [filter_config[0],]
        # load_layer = [[0, 2], [5, 7], [10, 12, 14], [17, 19, 21], None, None]
        load_layer = [None for x in range(6)]
        grad = [True, True, True, True, True, True]

        for i in range(len(filter_config)):
            # encoder architecture
            self.encoders.append(vgg_encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate, load_layer=load_layer[i], no_grad=grad[i]))

            # decoder architecture
            """if i == len(filter_config) - 1:
                decoder_filter_config[i] *= 3
                print(decoder_filter_config[i])"""
            self.decoders.append(vgg_decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        self.connect = nn.Conv2d(3*512, 512, kernel_size=1)

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat_in = x# .permute(0, 2, 3, 1).cpu().numpy()
        feat = x
        feat_list = []

        # encoder path, keep track of pooling indices and features size
        for t in range(3):
            if t > 0:
                # print(feat_in.shape)
                """feat_in = cv2.pyrDown(feat_in)
                feat_in = torch.FloatTensor(feat_in).cuda()"""
                feat_in = F.interpolate(feat_in, scale_factor=0.5, mode="bilinear")
                feat = feat_in
            for i in range(0, 4):
                (feat, ind), size = self.encoders[i](feat)
                if t == 0:
                    indices.append(ind)
                    unpool_sizes.append(size)
            feat_list.append(feat)

        feat_list[1] = F.interpolate(feat_list[1], size=feat_list[0].shape[2:], mode="bilinear")
        feat_list[2] = F.interpolate(feat_list[2], size=feat_list[0].shape[2:], mode="bilinear")
        feat = torch.cat(feat_list, dim=1)
        feat = self.connect(feat)
        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 4):
            feat = self.decoders[i](feat, indices[3 - i], unpool_sizes[3 - i])

        return F.softmax(self.classifier(feat), dim=1)


vgg_model = vgg(pretrained=True)
class vgg_encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5, no_grad=False, load_layer=None):
        super(vgg_encoder, self).__init__()
        # assert load_layer is not None

        conv1 = nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1)
        if load_layer is not None:
            conv1.weight.data.copy_(vgg_model.features[load_layer[0]].weight.data)
            conv1.bias.data.copy_(vgg_model.features[load_layer[0]].bias.data)
            conv1.weight.requires_grad = no_grad
            conv1.bias.requires_grad = no_grad
        layers = [conv1,
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            assert n_blocks <= 3
            if n_blocks == 3:
                conv3 = nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1)
                if load_layer is not None:
                    conv3.weight.data.copy_(vgg_model.features[load_layer[2]].weight.data)
                    conv3.bias.data.copy_(vgg_model.features[load_layer[2]].bias.data)
                    conv3.weight.requires_grad = no_grad
                    conv3.bias.requires_grad = no_grad
                layers += [conv3,
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True),
                           nn.Dropout(drop_rate)]
            elif n_blocks == 2:
                conv2 = nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1)
                if load_layer is not None:
                    conv2.weight.data.copy_(vgg_model.features[load_layer[1]].weight.data)
                    conv2.bias.data.copy_(vgg_model.features[load_layer[1]].bias.data)
                    conv2.weight.requires_grad = no_grad
                    conv2.bias.requires_grad = no_grad
                layers += [conv2,
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class vgg_decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5, no_grad=False, load_layer=None):
        super(vgg_decoder, self).__init__()

        conv1 = nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1)
        layers = [conv1,
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            assert n_blocks <= 3
            if n_blocks == 3:
                conv3 = nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1)
                layers += [conv3,
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True),
                           nn.Dropout(drop_rate)]
            elif n_blocks == 2:
                conv2 = nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1)
                layers += [conv2,
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)


if __name__ == "__main__":
    net = vgg_skip()
    # image = cv2.imread("./30001_gt.bmp")
    image = np.random.uniform(0, 1, size=(233, 233, 3))
    print("Image shape: ", image.shape)
    out = net(torch.FloatTensor(image).unsqueeze(0).permute(0, 3, 1, 2))
    print(out.shape)
    out = F.softmax(out, dim=1)
    print(out.shape)
    print(out)
    out = torch.argmax(out, dim=1, keepdim=True)
    print(out)
    print(out.shape)
