# foreground-segmentation
Final Project for 2019 PKU Digital Image Processing

## Raw GrabCut Algorithm
The performance of the GrabCut algorithm over the whole dataset with 15 iteration on each image is 0.9552, where the number of images with F-score lower than 0.80 is 39, higher than 0.95 is 781.

## FgSegNet
I've already implemented the PyTorch version FgSegNet_M put forward in paper "Foreground segmentation using convolutional neural networks for multiscale feature encoding" and now the model is training(Hope it works:).

Unfortunately, the model doesn't work, because it's facing overfitting problem, it nearly has no ability to generalize its performance to validation dataset. 

Now there are several choices to fix the problem:

- Shallowing the network architecture
- Use some data augmentation techniques (Underway...)
- Enlarge the dropout radio
