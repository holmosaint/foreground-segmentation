import os
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
aug_flip_horizontal = iaa.Sequential([iaa.Fliplr(0.5), iaa.CropToFixedSize(256, 256)]) # horizontally flip 50% of all images
aug_flip_vertical = iaa.Sequential([iaa.Flipud(0.2), iaa.CropToFixedSize(256, 256)]) # vertically flip 20% of all images
aug_crop = iaa.Sequential([
    sometimes(iaa.CropAndPad(
        percent=(-0.05, 0.1),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)
        )),
    iaa.CropToFixedSize(256, 256)])    # crop images by -5% to 10% of their height/width
aug_affine = iaa.Sequential([        
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        rotate=(-45, 45), # rotate by -45 to +45 degrees
        shear=(-16, 16), # shear by -16 to +16 degrees
        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )), 
    iaa.CropToFixedSize(256, 256)])
aug_invert = iaa.Sequential([iaa.Invert(0.8, per_channel=False), iaa.CropToFixedSize(256, 256)]) # invert

aug_list = [aug_flip_horizontal, aug_flip_vertical, aug_crop, aug_affine, aug_invert]

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


def Augmentation(image_list, mask_list):
    image_aug_list = list()
    mask_aug_list = list()
    # image_aug_list.extend(image_list)
    # mask_aug_list.extend(mask_list)
    for aug_method in aug_list:
        aug_image, aug_mask = aug_method(images=image_list, heatmaps=mask_list)
        image_aug_list.extend(aug_image)
        mask_aug_list.extend(aug_mask)
        
    return np.array(image_aug_list), np.array(mask_aug_list)


def DataGenerator(data_dir, gt_dir,augmentation=True):
    """The image in the data dir should be named as ID.jpg"""
    image_name_list = list()
    for _, _, image_name in os.walk(data_dir):
        image_name_list.extend(image_name)
    
    image_name_list = [x for x in image_name_list if x.endswith(".jpg")]
    image_name_list.sort()

    image_list = list()
    for image_name in image_name_list:
        image_path = os.path.join(data_dir, image_name)
        image = cv2.imread(image_path)
        image_list.append(image)
    
    """The mask in the gt dir should be named as ID_gt.bmp"""
    mask_name_list = list()
    for _, _, mask_name in os.walk(gt_dir):
        mask_name_list.extend(mask_name)

    mask_name_list = [x for x in mask_name_list if x.endswith(".bmp")]
    mask_name_list.sort()

    mask_list = list()
    for mask_name in mask_name_list:
        mask_path = os.path.join(gt_dir, mask_name)
        mask = cv2.imread(mask_path)
        mask_list.append(mask)
    
    if not augmentation:
        return np.array(image_list), np.array(mask_list)
    
    return Augmentation(image_list, mask_list)


if __name__ == "__main__":
    image = cv2.imread("./30001.jpg")
    mask = cv2.imread("./30001_gt.bmp")
    image_aug_flip_horizontal, mask_aug_flip_horizontal = aug_flip_horizontal(images=[image], heatmaps=[mask.astype(np.float32)])
    image_aug_flip_vertical, mask_aug_flip_vertical = aug_flip_vertical(images=[image], heatmaps=[mask.astype(np.float32)])
    image_aug_crop, mask_aug_crop = aug_crop(images=[image], heatmaps=[mask.astype(np.float32)])
    image_aug_affine, mask_aug_affine = aug_affine(images=[image], heatmaps=[mask.astype(np.float32)])
    image_aug_invert, mask_aug_invert = aug_invert(images=[image], heatmaps=[mask.astype(np.float32)])
    
    plt.xticks([])
    plt.subplot(2, 6, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2, 6, 2)
    plt.imshow(image_aug_flip_horizontal[0])
    plt.axis('off')
    plt.subplot(2, 6, 3)
    plt.imshow(image_aug_flip_vertical[0])
    plt.axis('off')
    plt.subplot(2, 6, 4)
    plt.imshow(image_aug_crop[0])
    plt.axis('off')
    plt.subplot(2, 6, 5)
    plt.imshow(image_aug_affine[0])
    plt.axis('off')
    plt.subplot(2, 6, 6)
    plt.imshow(image_aug_invert[0])    
    plt.axis('off')
    plt.subplot(2, 6, 7)
    plt.imshow(mask)
    plt.axis('off')
    plt.subplot(2, 6, 8)
    plt.imshow(mask_aug_flip_horizontal[0])
    plt.axis('off')
    plt.subplot(2, 6, 9)
    plt.imshow(mask_aug_flip_vertical[0])
    plt.axis('off')
    plt.subplot(2, 6, 10)
    plt.imshow(mask_aug_crop[0])
    plt.axis('off')
    plt.subplot(2, 6, 11)
    plt.imshow(mask_aug_affine[0])
    plt.axis('off')
    plt.subplot(2, 6, 12)
    plt.imshow(mask_aug_invert[0])
    plt.axis('off')
    plt.savefig("./aug_com.jpg", dpi=200)
