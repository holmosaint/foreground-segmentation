# foreground-segmentation
Final Project for 2019 PKU Digital Image Processing

## Raw GrabCut Algorithm
The performance of the GrabCut algorithm over the whole dataset with 15 iteration on each image is **0.9552**, where the number of images with F-score lower than 0.80 is 39, higher than 0.95 is 781.\

The biggest problem of GrabCut algorithm is the time consumption. The total time of GrabCut algorithm over the whole dataset is 26min 21s with 15 iterations for each picture. The avedrage time for one image is 1.581s.

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

After fine-tuning on the poor-performance images, and stimulated annealing for the best threshold after 30 iterations, the F-score comes to **0.81** with threshold 0.42. So I believe the problem is on the model architecture and image preprocessing. Many poor-performance images' foreground is darker than the background, but the majority of the dataset is the opposite. We should consider this problem carefully. 

## SegNet
I've implemented SegNet(based on others' code), the training behavior is quite good, but the final result isn't that good. The F score of on the test image is **0.73**. I believe that some of the error should go to the dataset itself. 

Now I'm going to try SegNet with data augmentation using in keras. 

The result of SegNet with data augmentation is quite good! We achieved **0.88** F-score which is currently the best result among our models. 

Then I implemented the dense CRF layer as the postprocessing procedure to polish the probability map. After 3 iteration on the map, we achieve **0.91** F-score on the whole dataset. 

The next step and the final step is to test the WaterShed algorithm, implement vgg-w-mask architecture and test GrabCut in one cut! Good luck!!!
