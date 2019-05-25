# foreground-segmentation
Final Project for 2019 PKU Digital Image Processing

## Raw GrabCut Algorithm
The performance of the GrabCut algorithm over the whole dataset with 15 iteration on each image is **0.9552**, where the number of images with F-score lower than 0.80 is 39, higher than 0.95 is 781.

## FgSegNet
I've already implemented the PyTorch version FgSegNet_M put forward in paper "Foreground segmentation using convolutional neural networks for multiscale feature encoding" and now the model is training(Hope it works:).

Unfortunately, the model doesn't work, because it's facing overfitting problem, it nearly has no ability to generalize its performance to validation dataset. 

Now there are several choices to fix the problem:

- Shallowing the network architecture
- Use some data augmentation techniques (Underway...)
- Enlarge the dropout radio

**Data Augmentation worked!**

By using the keras data augmentation tools, the binary entropy loss of both train and validation data set is under 0.2. The average F score over the whole dataset after training is **0.78**, when the threshold is 0.5. There are 47 images whose F score is lower than 0.4. The next step is to fine tuning the model on these images, with a smaller learning rate. 

The image augmentation configuration is as below:
```
train_datagen = ImageDataGenerator(rotation_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
Also note that the image is rescaled under factor 1/256.
```
