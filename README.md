# Kaggle

# Projects:

## [GPT]

### To do

- Parallelize the attention heads and compare the difference in running time
- Expand from 2 blocks to 6 blocks, and compare the loss and accuracy
- Create a ViT from scratch (essentially run Conv2d at the start, main structure remains untouched)

## [Leveraging autoencoders](/Leveraging%20autoencoders)

Leveraging autoencoders to utilize unlabelled data:

![modelDiagram](/Leveraging%20autoencoders/Images/modelDiagram.png)

![2](/Leveraging%20autoencoders/Images/predictionsWithReconstructions.png)


## [CNN for fashion mnist](https://github.com/Molten-Ice/Kaggle/blob/main/cnn-for-fashion-mnist.ipynb)

Uses Conv2d, MaxPool2d in a Tiny VGG style architecture.

Based on the below confusion matrix we can see the majority of the errors come predicting T-shirts, pullovers & Coats are shirts.
This is as the classes are very similar and so hard to distinguish (beyond human standard)


![TinyVGGConfusionMatrix](/FashionMNIST%20Tiny%20VGG%20inspired%20architecture/Images/TinyVGGConfusionMatrix.png) ![FashionMNIST](/FashionMNIST%20Tiny%20VGG%20inspired%20architecture/Images/FashionMNIST.png)

## [Modifying ResNet50 for MRI scans](https://github.com/Molten-Ice/Deep-Learning/tree/main/Brain%20MRIs(modified%20resnet50))

![BrainMRI](/Brain%20MRIs(modified%20resnet50)/Images/10MRIs.png)

## To Do:

- Creating heatmaps to show activations signifcance

- Feature visualization

- Create a Visual transformer from first principles
