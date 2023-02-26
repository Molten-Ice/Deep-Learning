# Kaggle

Following course: [learnpytorch.io](https://www.learnpytorch.io/)

### To do:

- Experience with CNNs on Fashion MNIST

- Create a basic auto-encoder & then a cnn autoencoder upscaling max pool

### Projects:

#### Autoencoder
Started 25/02/2023
Want to replicate MNIST wieth a high accuracy using a bottleneck of 10 nodes

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
V1
loss 0.1368 -> 0.0625
![V1](/Images/autoencoderV1.png)
