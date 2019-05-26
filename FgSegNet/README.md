## Usage
```
usage: main.py [-h] [--gpu GPU] [--encoder ENCODER]  
               [--train_image_dir TRAIN_IMAGE_DIR]  
               [--train_mask_dir TRAIN_MASK_DIR]  
               [--val_image_dir VAL_IMAGE_DIR] [--val_mask_dir VAL_MASK_DIR]  
               [--test_image_dir TEST_IMAGE_DIR]  
               [--test_mask_dir TEST_MASK_DIR] [--train TRAIN] [--test TEST]  
               [--epoch EPOCH] [--step_per_epoch STEP_PER_EPOCH]  
               [--batch BATCH] [--aug AUG]  

Foreground Segmentation for 2019 PKU Image Processing Course

optional arguments:  
  -h, --help            show this help message and exit  
  --gpu GPU             Whether to use gpu  
  --encoder ENCODER     Choose the architecture of the encoder network: vgg,  
                        resnet or segnet  
  --train_image_dir TRAIN_IMAGE_DIR  
  --train_mask_dir TRAIN_MASK_DIR  
  --val_image_dir VAL_IMAGE_DIR  
  --val_mask_dir VAL_MASK_DIR  
  --test_image_dir TEST_IMAGE_DIR  
  --test_mask_dir TEST_MASK_DIR  
  --train TRAIN  
  --test TEST  
  --epoch EPOCH  
  --step_per_epoch STEP_PER_EPOCH  
  --batch BATCH  
  --aug AUG  
```
### An Example
```
 python main.py --encoder segnet --gpu 1 --train 1 --aug keras --train_image_dir ./keras-aug/orderedImages/train/ --train_mask_dir ./keras-aug/orderedTruths/train/ --val_image_dir ./keras-aug/orderedImages/val/ --val_mask_dir ./keras-aug/orderedTruths/val
 ```
