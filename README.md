# Kaggle

Following course: [learnpytorch.io](https://www.learnpytorch.io/)

### To do:

- Experience with CNNs on Fashion MNIST

- Create a basic auto-encoder & then a cnn autoencoder upscaling max pool

### Projects:

#### Autoencoder
Started 25/02/2023
Want to replicate MNIST wieth a high accuracy using a bottleneck of 10 nodes

AudoencoderV1(
  (encoder): Sequential(
    (0): Linear(in_features=784, out_features=50, bias=True)
    (1): Linear(in_features=50, out_features=10, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=10, out_features=50, bias=True)
    (1): Linear(in_features=50, out_features=784, bias=True)
  )
)
V1
loss 0.1368 -> 0.0625
![V1](autoencoderv1.png)
