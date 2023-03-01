# Creating a CNN( Tiny VGG inspired) for classification in FashionMNIST

## Development

The model is inspired by Tiny VGG inspired in the sense, we will repeat the following. 

In each block we'll have 2 convolutional layers which preserve size, followed by a max pool layer which half width and length. We'll repeat this process 3 times going from

28x28 -> 14x14 -> 7x7 -> 3x3

before flattening the model and feating into into a linear layer, followed by a classification layer.

As this is a classification problem over multiple classes I thought the cross entropy loss function most suitable.

Side note: The ReLU activation function will be interspersed between almost every layer. Coming from a mathematians background I assumed (wrongly) that sigmoid would be supreme. But so far it seems to perform worse in almost any case. For this reason I'm going to be sticking to ReLU(or leaky ReLU or even GeLU :O) , unless the situation demands I use sigmoid.

### V1

Achieves 83% using a Linear style architecture, namely

```
FashionMNISTModelV0(
  (layer_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
  )
)
```

### V2

Created a TinyVGG inspired architecture but it was not working.
(can't remember why)

The below was for a different problem, the autoencoder.
I believe this was as the gradients were spiralling out of control (exploding gradient problem), my solution was to use batch normalization. Although in hindsight I think I went abit overkill on it ;)

```
FashionMNISTModelV2(
  (block_1): Sequential(
    (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=490, out_features=10, bias=True)
  )
)
```

### V3

7 epochs into training it gave the best results of:

```
Train loss: 0.31982 | Train accuracy: 88.29%
Test loss: 0.33969 | Test accuracy: 88.17%
```

By the end of the 10 epochs the train accuracy has improved marginally, but the test accuracy had dropped alot, which is a sign of overfitting started to happen

```
Train loss: 0.29356 | Train accuracy: 89.08%
Test loss: 0.37845 | Test accuracy: 85.94%
```


Which improved the results from the linear network by over 10%!!

```
Train loss: 3.69213 | Train accuracy: 75.01%
Test loss: 2.73955 | Test accuracy: 77.70%
```


Notes: The method for calculating the loss here was inside the inner batch loop. This means the loss gets added over different stages of the training, as it learns for each batch. This means we don't have to use additionally computing power to find the training loss for each epoch. However, it is not strictly the best representation and so I've now refrained from doing it in my future models.

## Results Analysis

![/Images/TinyVGGConfusionMatrix]

From the above confusion matrix we can see the majority of the error term is coming from the model incorrectly predicting coats, dresses, pullovers and T-shirts as shirts.

Looking at examples from the dataset, I personally struggle to tell the difference for many of the examples as they are so similar. A potential cause for this could be incorrect labelling for the dataset. Perhaps shirt should've been split into separate categories.

The confusion matrix is a lovely way of representing the results as I can clearly see what's going on. e.g. Ankle boots and Bags are almost always predicted correctly. And most of the loss comes from a small subset of similar looking items which blur the class boundaries.