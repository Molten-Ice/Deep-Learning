# Leveraging an autoencoder to utilize unlabelled data

[Autoencoder Development](https://github.com/Molten-Ice/Kaggle/tree/main/Leveraging%20autoencoders#autoencoder-development)

[Adding Neural network for predictions](https://github.com/Molten-Ice/Kaggle/tree/main/Leveraging%20autoencoders#adding-neural-network-for-predictions)


## Autoencoder Development
Started 25/02/2023
Want to replicate MNIST with a high accuracy using a bottleneck of 10 nodes

### V1
```
self.encoder = nn.Sequential(
            nn.Linear(in_features=784,
                      out_features=hidden_units),
            nn.Linear(in_features = hidden_units,
                      out_features = bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=bottleneck,
                      out_features=hidden_units),
            nn.Linear(in_features = hidden_units,
                      out_features = 784)
        )
```
Trained over 1000 batches of 32 elements & 5 epochs.

loss 0.1368 -> 0.0625
![V1](/Leveraging%20autoencoders/Images/autoencoderV1.png)

#### V2

Adding non-linearity 
```
self.encoder = nn.Sequential(
            nn.Linear(in_features=784,
                      out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features = hidden_units,
                      out_features = bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=bottleneck,
                      out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features = hidden_units,
                      out_features = 784),
            nn.Sigmoid()
        )
```
Trained over 1000 batches of 32 elements & 5 epochs
Loss 0.2375->0.0746
![V2](/Leveraging%20autoencoders/Images/autoencoderV2.png)

#### V3

- Changing nin-linearity to focus on ReLU rather than sigmoid.
- Creating size of nn massively, now 1500, 1000, 500, 10 nodes in each layer in the code (and symmetrically for the decode)
- Added a learning rate scheduler, halfing the lr every 50 epochs
```
AudoencoderV3(
  (encoder): Sequential(
    (0): Linear(in_features=784, out_features=1500, bias=True)
    (1): ReLU()
    (2): Linear(in_features=1500, out_features=1000, bias=True)
    (3): ReLU()
    (4): Linear(in_features=1000, out_features=500, bias=True)
    (5): ReLU()
    (6): Linear(in_features=500, out_features=20, bias=True)
    (7): Sigmoid()
  )
  (decoder): Sequential(
    (0): Linear(in_features=20, out_features=500, bias=True)
    (1): ReLU()
    (2): Linear(in_features=500, out_features=1000, bias=True)
    (3): ReLU()
    (4): Linear(in_features=1000, out_features=1500, bias=True)
    (5): ReLU()
    (6): Linear(in_features=1500, out_features=784, bias=True)
  )
)
```

Plateaued 40 epochs in, reaching 0.0673 (i.e the same as the old models).
Didn't improve over the next 950 epochs.

![V3](/Leveraging%20autoencoders/Images/autoencoderV3.png)

#### V4

- Adding batchnorm
- Reshaping neural network (making it much much smaller)
- Changed Adam parameters
- Decreased lr decay

``` 
      l1 = 128
        l2 = 64
        l3 = 32
        bottleneck = 16
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=784,
                      out_features=l1),
            nn.BatchNorm1d(num_features = l1, affine = False),
            nn.Linear(in_features = l1,
                      out_features = l2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = l2, affine = False),
            nn.Linear(in_features = l2,
                      out_features = l3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = l3, affine = False),
            nn.Linear(in_features = l3,
                      out_features = bottleneck),
            nn.BatchNorm1d(num_features = bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=bottleneck,
                      out_features=l3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = l3, affine = False),
            nn.Linear(in_features=l3,
                      out_features=l2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = l2, affine = False),
            nn.Linear(in_features = l2,
                      out_features = l1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = l1, affine = False),
            nn.Linear(in_features = l1,
                      out_features = 784),
            nn.ReLU()
        )
```

![V4](/Leveraging%20autoencoders/Images/autoencoderV4.png)

#### V5

Currently the issue I am facing is exploding loss.
30 epocds in it is down to 0.0636, then goes
0.0636->0.173->0.14432->0.1317->39.58->149.78->0.8528->0.15

I think the reason of this is due to some numerical instability (close to /0) which causes huge updates in backpropagation.


Adding normalizion to the inputs, instead of just / 255. the input.
```
self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features = 784, affine = False),
            nn.Linear(in_features=784,
```

![V5](/Leveraging%20autoencoders/Images/autoencoderV5.png)

#### V6

Insanely good predictions (using bottleneck of 10), but awful loss. (literally reaches 1400, then goes to a 18 digit long number). 
I think this is as its predicting the correct pixels but the values it finds for them is unbounded (as the last function is ReLU)

Epoch: 181 | Train loss: 874721216.0000 | Test loss: 372588722165645312.0000 | Learning rate: 0.0125

![V6](/Leveraging%20autoencoders/Images/autoencoderV6.png)

#### V7 (Working)

The issue was the output's from the last layer of the Neurual network were unbounded. The last ReLU layer meant there was no upper bound, which caused an exploding gradient issue (I think).
To combat this I retricted the last layer to the range [0,255], with the results of the two methods shown below.

After 250 epochs
ReLU(with clamp) gives loss: 1154.6481
Sigmoid(with x 255) gives loss: 2036.6487

![V7](/Leveraging%20autoencoders/Images/autoencoderV7.png)


## Adding Neural network for predictions

Using the encoder part of the encoder with frozen parameters, with a neural network added on the end to generate predictions

#### V1
I initally thought it was working very well, but had accidentally typed (train_acc) instead of (test_acc) so it was repeating the same result twice.
Very good train accuracy and loss, but not transferring over to the test dataset.
Currently using 1000 elements with a batch size of 32

E:5, b:0 |Train loss: 0.2339, acc:96.0938 | Test loss: 0.5528, acc: 11.7188 | lr: 0.1

Realized my error when I checked the values and the predictions were awful

```
SuffixNN(
  (encoder): Sequential(
    (0): BatchNorm1d(784, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (1): Linear(in_features=784, out_features=128, bias=True)
    (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (6): Linear(in_features=64, out_features=32, bias=True)
    (7): ReLU()
    (8): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (9): Linear(in_features=32, out_features=16, bias=True)
    (10): ReLU()
    (11): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (12): Linear(in_features=16, out_features=10, bias=True)
    (13): ReLU()
    (14): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  )
  (layers): Sequential(
    (0): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (1): Linear(in_features=10, out_features=50, bias=True)
    (2): ReLU()
    (3): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (4): Linear(in_features=50, out_features=10, bias=True)
  )
)
add Code
```

#### V2

My model is performing well on training data but badly on unseen data

- Trying Dropout

After 1 epoch this was the result:
E:0, b:15 |Train loss: 0.5141, acc:84.0332 | Test loss: 0.5743, acc: 9.0332 | lr: 0.1
The train and test losses are similar (both having signficantly decreased since their first values of 1.9),
but test accuracy didn't decrease.
This led me to find the error:

```
test_pred_labels = torch.argmax(y_t_logits, dim=1)
``` 
should instead have 
```y_test_logits``` 
inside

Explaining why the test accuracy was no better than random

Also changed model architecutre to:
```
l1 = 50
l2 = 50      

self.layers = nn.Sequential(
nn.BatchNorm1d(num_features = 10, affine = False),
nn.Linear(in_features=10,
out_features=l1),
nn.BatchNorm1d(num_features = l1, affine = False),
nn.ReLU(),
nn.Dropout(p=0.2),
nn.Linear(in_features = l1,
out_features = l1),
nn.BatchNorm1d(num_features = l2, affine = False),
nn.ReLU(),
nn.Dropout(p=0.2),
nn.Linear(in_features = l2,
out_features = 10))
```

Order of Linear, normalization, activation and dropout layers seems to be greatly debating online. Will need to do some furhter research into it

To try: L1, L2 regularization

#### V3

Model working correctly.
Reduced the NN to only a have a single layer of 50 nodes.
Giving the following loss and accuracy 

E:9, b:15 |Train loss: 0.3213, acc:90.1367 | Test loss: 0.3504, acc: 89.0137 | lr: 0.0125

Full NN design:
```
SuffixNN(
  (encoder): Sequential(
    (0): BatchNorm1d(784, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (1): Linear(in_features=784, out_features=128, bias=True)
    (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (6): Linear(in_features=64, out_features=32, bias=True)
    (7): ReLU()
    (8): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (9): Linear(in_features=32, out_features=16, bias=True)
    (10): ReLU()
    (11): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (12): Linear(in_features=16, out_features=10, bias=True)
    (13): ReLU()
    (14): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  )
  (layers): Sequential(
    (0): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (1): Linear(in_features=10, out_features=50, bias=True)
    (2): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.2, inplace=False)
    (5): Linear(in_features=50, out_features=10, bias=True)
  )
)
```

Normal predictions
![1](/Leveraging%20autoencoders/Images/predictions.png)

Predictions with reconstructions from autoencoder
![2](/Leveraging%20autoencoders/Images/predictionsWithReconstructions.png)

Wrong predictions from the model, with their reconstructions
![3](/Leveraging%20autoencoders/Images/wrongExamples.png)

From this we can see the majority of the errors are caused through the error of the autoencoder, rather than the suffix neural network after it. I can attempt to optimize this by using a convolutional artitecture.

#### V4

Reducing training data
