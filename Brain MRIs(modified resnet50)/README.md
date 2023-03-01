
# Modifying a pretrained resnet50 to use on a custom dataset


This dataset contained a collection of MRI images for the detection of Brain Tumors

[Kaggle dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?datasetId=165566)


[10Images](/Images/10MRIs.png)

## Development


This was the first dataset I approached which was in a none standard form, and would have to refactor into pytorch useable data structures.

### The plan:

raw data -> torch.utils.data.Dataset -> torch.utils.data.DataLoader

The dataset has 98 "no" images and 155 "yes" images.
Which look like the following
[Scan](/Images/MRI1.png)

The data was all over the place, different numbers of channels (1-4), different sizes, etc.

### Issues:

- The first issue I came across was data with 4 colour channels, as this accounted for only 5 of the 250 images I choose to ignore this data entirely

- The second (major) issue was PIL.Image turns grayscale images into a 2d tensor e.g. (224, 224), even if I artifically unsqueeze() the array into number and then convert back into a PIL.Image object. I got around this by converting it into a numpy array, stacking 3 ontop of each other to give a tensor of shape (224, 224, 3) and then convert back to a PIL.Image object.

### Notes:

 - transforms.Resize((224, 224)) or transforms.ToTensor() is swapping the order of the Tensor to (3, 224, 224), placing the colour channel in the correct spot for pytorch models. (this is does implicitly rather than me permuting the dimensions)

 - transforms.Grayscale(num_output_channels=3) is expanding out all of the images into a standard 3 colour channels repesentation




### Improvements I could make: 
- I should balance the number of "yes" and "no" examples between the train and test data (and use a different accuracy metric as the dataset is imbalanced)

- I could use further data augmentation to generate multiple test images from a single train example


### Code:

Loading Resnet50 model with pretrained weights
```
weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights).to(device)
```

Freeze all layers apart from final fully connected layer

```
layers_to_freeze = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
for layer in layers_to_freeze:
    for param in layer.parameters():
        param.requires_grad = False
```

```
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=2048, 
                    out_features=1, # same number of output units as our number of classes
                    bias=True),
    nn.Sigmoid()).to(device)
```

Using 23,000,000 frozen parameters & 2,000 learnable ones!!


### Model results:

Running attempt 1, Adam with 0.1 lr for 10 epochs.

[Image1](/Images/LossGraph)

[Image2](/Images/AccuracyGraph)

woahhh!

Something is not going right here, it seems to be almost "random"

One thing that instantly draws my eye is that for the first 3 plot points (15ish batches, i.e. 1 epoch) the accuracy rapidly climbs.

My thoughts: decrease the learning rate and train over 1/2 epochs to see what happens


### Full Model:

```
========================================================================================================================
Layer (type (var_name))                  Input Shape          Output Shape         Param #              Trainable
========================================================================================================================
ResNet (ResNet)                          [8, 3, 224, 224]     [8, 1]               --                   Partial
├─Conv2d (conv1)                         [8, 3, 224, 224]     [8, 64, 112, 112]    (9,408)              False
├─BatchNorm2d (bn1)                      [8, 64, 112, 112]    [8, 64, 112, 112]    (128)                False
├─ReLU (relu)                            [8, 64, 112, 112]    [8, 64, 112, 112]    --                   --
├─MaxPool2d (maxpool)                    [8, 64, 112, 112]    [8, 64, 56, 56]      --                   --
├─Sequential (layer1)                    [8, 64, 56, 56]      [8, 256, 56, 56]     --                   False
│    └─Bottleneck (0)                    [8, 64, 56, 56]      [8, 256, 56, 56]     --                   False
│    │    └─Conv2d (conv1)               [8, 64, 56, 56]      [8, 64, 56, 56]      (4,096)              False
│    │    └─BatchNorm2d (bn1)            [8, 64, 56, 56]      [8, 64, 56, 56]      (128)                False
│    │    └─ReLU (relu)                  [8, 64, 56, 56]      [8, 64, 56, 56]      --                   --
│    │    └─Conv2d (conv2)               [8, 64, 56, 56]      [8, 64, 56, 56]      (36,864)             False
│    │    └─BatchNorm2d (bn2)            [8, 64, 56, 56]      [8, 64, 56, 56]      (128)                False
│    │    └─ReLU (relu)                  [8, 64, 56, 56]      [8, 64, 56, 56]      --                   --
│    │    └─Conv2d (conv3)               [8, 64, 56, 56]      [8, 256, 56, 56]     (16,384)             False
│    │    └─BatchNorm2d (bn3)            [8, 256, 56, 56]     [8, 256, 56, 56]     (512)                False
│    │    └─Sequential (downsample)      [8, 64, 56, 56]      [8, 256, 56, 56]     (16,896)             False
│    │    └─ReLU (relu)                  [8, 256, 56, 56]     [8, 256, 56, 56]     --                   --
│    └─Bottleneck (1)                    [8, 256, 56, 56]     [8, 256, 56, 56]     --                   False
│    │    └─Conv2d (conv1)               [8, 256, 56, 56]     [8, 64, 56, 56]      (16,384)             False
│    │    └─BatchNorm2d (bn1)            [8, 64, 56, 56]      [8, 64, 56, 56]      (128)                False
│    │    └─ReLU (relu)                  [8, 64, 56, 56]      [8, 64, 56, 56]      --                   --
│    │    └─Conv2d (conv2)               [8, 64, 56, 56]      [8, 64, 56, 56]      (36,864)             False
│    │    └─BatchNorm2d (bn2)            [8, 64, 56, 56]      [8, 64, 56, 56]      (128)                False
│    │    └─ReLU (relu)                  [8, 64, 56, 56]      [8, 64, 56, 56]      --                   --
│    │    └─Conv2d (conv3)               [8, 64, 56, 56]      [8, 256, 56, 56]     (16,384)             False
│    │    └─BatchNorm2d (bn3)            [8, 256, 56, 56]     [8, 256, 56, 56]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 56, 56]     [8, 256, 56, 56]     --                   --
│    └─Bottleneck (2)                    [8, 256, 56, 56]     [8, 256, 56, 56]     --                   False
│    │    └─Conv2d (conv1)               [8, 256, 56, 56]     [8, 64, 56, 56]      (16,384)             False
│    │    └─BatchNorm2d (bn1)            [8, 64, 56, 56]      [8, 64, 56, 56]      (128)                False
│    │    └─ReLU (relu)                  [8, 64, 56, 56]      [8, 64, 56, 56]      --                   --
│    │    └─Conv2d (conv2)               [8, 64, 56, 56]      [8, 64, 56, 56]      (36,864)             False
│    │    └─BatchNorm2d (bn2)            [8, 64, 56, 56]      [8, 64, 56, 56]      (128)                False
│    │    └─ReLU (relu)                  [8, 64, 56, 56]      [8, 64, 56, 56]      --                   --
│    │    └─Conv2d (conv3)               [8, 64, 56, 56]      [8, 256, 56, 56]     (16,384)             False
│    │    └─BatchNorm2d (bn3)            [8, 256, 56, 56]     [8, 256, 56, 56]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 56, 56]     [8, 256, 56, 56]     --                   --
├─Sequential (layer2)                    [8, 256, 56, 56]     [8, 512, 28, 28]     --                   False
│    └─Bottleneck (0)                    [8, 256, 56, 56]     [8, 512, 28, 28]     --                   False
│    │    └─Conv2d (conv1)               [8, 256, 56, 56]     [8, 128, 56, 56]     (32,768)             False
│    │    └─BatchNorm2d (bn1)            [8, 128, 56, 56]     [8, 128, 56, 56]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 56, 56]     [8, 128, 56, 56]     --                   --
│    │    └─Conv2d (conv2)               [8, 128, 56, 56]     [8, 128, 28, 28]     (147,456)            False
│    │    └─BatchNorm2d (bn2)            [8, 128, 28, 28]     [8, 128, 28, 28]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 28, 28]     [8, 128, 28, 28]     --                   --
│    │    └─Conv2d (conv3)               [8, 128, 28, 28]     [8, 512, 28, 28]     (65,536)             False
│    │    └─BatchNorm2d (bn3)            [8, 512, 28, 28]     [8, 512, 28, 28]     (1,024)              False
│    │    └─Sequential (downsample)      [8, 256, 56, 56]     [8, 512, 28, 28]     (132,096)            False
│    │    └─ReLU (relu)                  [8, 512, 28, 28]     [8, 512, 28, 28]     --                   --
│    └─Bottleneck (1)                    [8, 512, 28, 28]     [8, 512, 28, 28]     --                   False
│    │    └─Conv2d (conv1)               [8, 512, 28, 28]     [8, 128, 28, 28]     (65,536)             False
│    │    └─BatchNorm2d (bn1)            [8, 128, 28, 28]     [8, 128, 28, 28]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 28, 28]     [8, 128, 28, 28]     --                   --
│    │    └─Conv2d (conv2)               [8, 128, 28, 28]     [8, 128, 28, 28]     (147,456)            False
│    │    └─BatchNorm2d (bn2)            [8, 128, 28, 28]     [8, 128, 28, 28]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 28, 28]     [8, 128, 28, 28]     --                   --
│    │    └─Conv2d (conv3)               [8, 128, 28, 28]     [8, 512, 28, 28]     (65,536)             False
│    │    └─BatchNorm2d (bn3)            [8, 512, 28, 28]     [8, 512, 28, 28]     (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 28, 28]     [8, 512, 28, 28]     --                   --
│    └─Bottleneck (2)                    [8, 512, 28, 28]     [8, 512, 28, 28]     --                   False
│    │    └─Conv2d (conv1)               [8, 512, 28, 28]     [8, 128, 28, 28]     (65,536)             False
│    │    └─BatchNorm2d (bn1)            [8, 128, 28, 28]     [8, 128, 28, 28]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 28, 28]     [8, 128, 28, 28]     --                   --
│    │    └─Conv2d (conv2)               [8, 128, 28, 28]     [8, 128, 28, 28]     (147,456)            False
│    │    └─BatchNorm2d (bn2)            [8, 128, 28, 28]     [8, 128, 28, 28]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 28, 28]     [8, 128, 28, 28]     --                   --
│    │    └─Conv2d (conv3)               [8, 128, 28, 28]     [8, 512, 28, 28]     (65,536)             False
│    │    └─BatchNorm2d (bn3)            [8, 512, 28, 28]     [8, 512, 28, 28]     (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 28, 28]     [8, 512, 28, 28]     --                   --
│    └─Bottleneck (3)                    [8, 512, 28, 28]     [8, 512, 28, 28]     --                   False
│    │    └─Conv2d (conv1)               [8, 512, 28, 28]     [8, 128, 28, 28]     (65,536)             False
│    │    └─BatchNorm2d (bn1)            [8, 128, 28, 28]     [8, 128, 28, 28]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 28, 28]     [8, 128, 28, 28]     --                   --
│    │    └─Conv2d (conv2)               [8, 128, 28, 28]     [8, 128, 28, 28]     (147,456)            False
│    │    └─BatchNorm2d (bn2)            [8, 128, 28, 28]     [8, 128, 28, 28]     (256)                False
│    │    └─ReLU (relu)                  [8, 128, 28, 28]     [8, 128, 28, 28]     --                   --
│    │    └─Conv2d (conv3)               [8, 128, 28, 28]     [8, 512, 28, 28]     (65,536)             False
│    │    └─BatchNorm2d (bn3)            [8, 512, 28, 28]     [8, 512, 28, 28]     (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 28, 28]     [8, 512, 28, 28]     --                   --
├─Sequential (layer3)                    [8, 512, 28, 28]     [8, 1024, 14, 14]    --                   False
│    └─Bottleneck (0)                    [8, 512, 28, 28]     [8, 1024, 14, 14]    --                   False
│    │    └─Conv2d (conv1)               [8, 512, 28, 28]     [8, 256, 28, 28]     (131,072)            False
│    │    └─BatchNorm2d (bn1)            [8, 256, 28, 28]     [8, 256, 28, 28]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 28, 28]     [8, 256, 28, 28]     --                   --
│    │    └─Conv2d (conv2)               [8, 256, 28, 28]     [8, 256, 14, 14]     (589,824)            False
│    │    └─BatchNorm2d (bn2)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv3)               [8, 256, 14, 14]     [8, 1024, 14, 14]    (262,144)            False
│    │    └─BatchNorm2d (bn3)            [8, 1024, 14, 14]    [8, 1024, 14, 14]    (2,048)              False
│    │    └─Sequential (downsample)      [8, 512, 28, 28]     [8, 1024, 14, 14]    (526,336)            False
│    │    └─ReLU (relu)                  [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   --
│    └─Bottleneck (1)                    [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   False
│    │    └─Conv2d (conv1)               [8, 1024, 14, 14]    [8, 256, 14, 14]     (262,144)            False
│    │    └─BatchNorm2d (bn1)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv2)               [8, 256, 14, 14]     [8, 256, 14, 14]     (589,824)            False
│    │    └─BatchNorm2d (bn2)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv3)               [8, 256, 14, 14]     [8, 1024, 14, 14]    (262,144)            False
│    │    └─BatchNorm2d (bn3)            [8, 1024, 14, 14]    [8, 1024, 14, 14]    (2,048)              False
│    │    └─ReLU (relu)                  [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   --
│    └─Bottleneck (2)                    [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   False
│    │    └─Conv2d (conv1)               [8, 1024, 14, 14]    [8, 256, 14, 14]     (262,144)            False
│    │    └─BatchNorm2d (bn1)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv2)               [8, 256, 14, 14]     [8, 256, 14, 14]     (589,824)            False
│    │    └─BatchNorm2d (bn2)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv3)               [8, 256, 14, 14]     [8, 1024, 14, 14]    (262,144)            False
│    │    └─BatchNorm2d (bn3)            [8, 1024, 14, 14]    [8, 1024, 14, 14]    (2,048)              False
│    │    └─ReLU (relu)                  [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   --
│    └─Bottleneck (3)                    [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   False
│    │    └─Conv2d (conv1)               [8, 1024, 14, 14]    [8, 256, 14, 14]     (262,144)            False
│    │    └─BatchNorm2d (bn1)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv2)               [8, 256, 14, 14]     [8, 256, 14, 14]     (589,824)            False
│    │    └─BatchNorm2d (bn2)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv3)               [8, 256, 14, 14]     [8, 1024, 14, 14]    (262,144)            False
│    │    └─BatchNorm2d (bn3)            [8, 1024, 14, 14]    [8, 1024, 14, 14]    (2,048)              False
│    │    └─ReLU (relu)                  [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   --
│    └─Bottleneck (4)                    [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   False
│    │    └─Conv2d (conv1)               [8, 1024, 14, 14]    [8, 256, 14, 14]     (262,144)            False
│    │    └─BatchNorm2d (bn1)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv2)               [8, 256, 14, 14]     [8, 256, 14, 14]     (589,824)            False
│    │    └─BatchNorm2d (bn2)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv3)               [8, 256, 14, 14]     [8, 1024, 14, 14]    (262,144)            False
│    │    └─BatchNorm2d (bn3)            [8, 1024, 14, 14]    [8, 1024, 14, 14]    (2,048)              False
│    │    └─ReLU (relu)                  [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   --
│    └─Bottleneck (5)                    [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   False
│    │    └─Conv2d (conv1)               [8, 1024, 14, 14]    [8, 256, 14, 14]     (262,144)            False
│    │    └─BatchNorm2d (bn1)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv2)               [8, 256, 14, 14]     [8, 256, 14, 14]     (589,824)            False
│    │    └─BatchNorm2d (bn2)            [8, 256, 14, 14]     [8, 256, 14, 14]     (512)                False
│    │    └─ReLU (relu)                  [8, 256, 14, 14]     [8, 256, 14, 14]     --                   --
│    │    └─Conv2d (conv3)               [8, 256, 14, 14]     [8, 1024, 14, 14]    (262,144)            False
│    │    └─BatchNorm2d (bn3)            [8, 1024, 14, 14]    [8, 1024, 14, 14]    (2,048)              False
│    │    └─ReLU (relu)                  [8, 1024, 14, 14]    [8, 1024, 14, 14]    --                   --
├─Sequential (layer4)                    [8, 1024, 14, 14]    [8, 2048, 7, 7]      --                   False
│    └─Bottleneck (0)                    [8, 1024, 14, 14]    [8, 2048, 7, 7]      --                   False
│    │    └─Conv2d (conv1)               [8, 1024, 14, 14]    [8, 512, 14, 14]     (524,288)            False
│    │    └─BatchNorm2d (bn1)            [8, 512, 14, 14]     [8, 512, 14, 14]     (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 14, 14]     [8, 512, 14, 14]     --                   --
│    │    └─Conv2d (conv2)               [8, 512, 14, 14]     [8, 512, 7, 7]       (2,359,296)          False
│    │    └─BatchNorm2d (bn2)            [8, 512, 7, 7]       [8, 512, 7, 7]       (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 7, 7]       [8, 512, 7, 7]       --                   --
│    │    └─Conv2d (conv3)               [8, 512, 7, 7]       [8, 2048, 7, 7]      (1,048,576)          False
│    │    └─BatchNorm2d (bn3)            [8, 2048, 7, 7]      [8, 2048, 7, 7]      (4,096)              False
│    │    └─Sequential (downsample)      [8, 1024, 14, 14]    [8, 2048, 7, 7]      (2,101,248)          False
│    │    └─ReLU (relu)                  [8, 2048, 7, 7]      [8, 2048, 7, 7]      --                   --
│    └─Bottleneck (1)                    [8, 2048, 7, 7]      [8, 2048, 7, 7]      --                   False
│    │    └─Conv2d (conv1)               [8, 2048, 7, 7]      [8, 512, 7, 7]       (1,048,576)          False
│    │    └─BatchNorm2d (bn1)            [8, 512, 7, 7]       [8, 512, 7, 7]       (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 7, 7]       [8, 512, 7, 7]       --                   --
│    │    └─Conv2d (conv2)               [8, 512, 7, 7]       [8, 512, 7, 7]       (2,359,296)          False
│    │    └─BatchNorm2d (bn2)            [8, 512, 7, 7]       [8, 512, 7, 7]       (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 7, 7]       [8, 512, 7, 7]       --                   --
│    │    └─Conv2d (conv3)               [8, 512, 7, 7]       [8, 2048, 7, 7]      (1,048,576)          False
│    │    └─BatchNorm2d (bn3)            [8, 2048, 7, 7]      [8, 2048, 7, 7]      (4,096)              False
│    │    └─ReLU (relu)                  [8, 2048, 7, 7]      [8, 2048, 7, 7]      --                   --
│    └─Bottleneck (2)                    [8, 2048, 7, 7]      [8, 2048, 7, 7]      --                   False
│    │    └─Conv2d (conv1)               [8, 2048, 7, 7]      [8, 512, 7, 7]       (1,048,576)          False
│    │    └─BatchNorm2d (bn1)            [8, 512, 7, 7]       [8, 512, 7, 7]       (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 7, 7]       [8, 512, 7, 7]       --                   --
│    │    └─Conv2d (conv2)               [8, 512, 7, 7]       [8, 512, 7, 7]       (2,359,296)          False
│    │    └─BatchNorm2d (bn2)            [8, 512, 7, 7]       [8, 512, 7, 7]       (1,024)              False
│    │    └─ReLU (relu)                  [8, 512, 7, 7]       [8, 512, 7, 7]       --                   --
│    │    └─Conv2d (conv3)               [8, 512, 7, 7]       [8, 2048, 7, 7]      (1,048,576)          False
│    │    └─BatchNorm2d (bn3)            [8, 2048, 7, 7]      [8, 2048, 7, 7]      (4,096)              False
│    │    └─ReLU (relu)                  [8, 2048, 7, 7]      [8, 2048, 7, 7]      --                   --
├─AdaptiveAvgPool2d (avgpool)            [8, 2048, 7, 7]      [8, 2048, 1, 1]      --                   --
├─Sequential (fc)                        [8, 2048]            [8, 1]               --                   True
│    └─Dropout (0)                       [8, 2048]            [8, 2048]            --                   --
│    └─Linear (1)                        [8, 2048]            [8, 1]               2,049                True
│    └─Sigmoid (2)                       [8, 1]               [8, 1]               --                   --
========================================================================================================================
Total params: 23,510,081
Trainable params: 2,049
Non-trainable params: 23,508,032
Total mult-adds (G): 32.70
========================================================================================================================
Input size (MB): 4.82
Forward/backward pass size (MB): 1422.59
Params size (MB): 94.04
Estimated Total Size (MB): 1521.45
========================================================================================================================

```