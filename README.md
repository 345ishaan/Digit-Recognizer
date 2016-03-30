# Digit-Recognizer
Hi, 

This git project clone will help you in setting up all the Data, Convolution and Pooling Layers to train your system to recognize a digit character using MNIST Database downloaded from Kaggle.

One thing worth noting is that the net is trained with images having "black" as background with normalized pixel value of "1" and digits seen visually as white represented by "0"

If you go ahead with loading an image with vice-versa pixel values , your label probabilities will not be deterministic.

The solution to this is to load the image and subtract "1.0" from each pixel value , hence inverting the color of the image. I will try releasing this in the next commit. But just for a note, the solution is:-

    Load Image
    Do :- Image= 1.0 - Image
