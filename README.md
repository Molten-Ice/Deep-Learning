# Kaggle

Following course: [learnpytorch.io](https://www.learnpytorch.io/)

# To do:

- Experience with CNNs on Fashion MNIST

- Create a basic auto-encoder & then a cnn autoencoder upscaling max pool

# Projects:

## Autoencoder
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
![V1](/Images/autoencoderV1.png)

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
![V2](/Images/autoencoderV2.png)

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

![V3](/Images/autoencoderV3.png)

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

![V4](/Images/autoencoderV4.png)

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

![V5](/Images/autoencoderV5.png)

#### V6

Insanely good predictions (using bottleneck of 10), but awful loss. (literally reaches 1400, then goes to a 18 digit long number). 
I think this is as its predicting the correct pixels but the values it finds for them is unbounded (as the last function is ReLU)

Epoch: 181 | Train loss: 874721216.0000 | Test loss: 372588722165645312.0000 | Learning rate: 0.0125

![V6](/Images/autoencoderV6.png)

#### V7 (Working)

The issue was the output's from the last layer of the Neurual network were unbounded. The last ReLU layer meant there was no upper bound, which caused an exploding gradient issue (I think).
To combat this I retricted the last layer to the range [0,255], with the results of the two methods shown below.

After 250 epochs
# ReLU(with clamp) gives loss: 1154.6481
# Sigmoid(with x 255) gives loss: 2036.6487

![V7](/Images/autoencoderV7.png)
