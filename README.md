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
