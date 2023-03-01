
# Modifying a pretrained resnet50 to use on a custom dataset


This dataset contained a collection of MRI images for the detection of Brain Tumors

[Kaggle dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?datasetId=165566)


[10Images](/Images/10MRIs.png)

## Development


This was the first dataset I approached which was in a none standard form, and would have to refactor into pytorch useable data structures.

The plan:

raw data -> torch.utils.data.Dataset -> torch.utils.data.DataLoader

The dataset has 98 "no" images and 155 "yes" images.
Which look like the following
[Scan](/Images/MRI1.png)

The data was all over the place, different numbers of channels (1-4), different sizes, etc.

Issues:

- The first issue I came across was data with 4 colour channels, as this accounted for only 5 of the 250 images I choose to ignore this data entirely

- The second (major) issue was PIL.Image turns grayscale images into a 2d tensor e.g. (224, 224), even if I artifically unsqueeze() the array into number and then convert back into a PIL.Image object. I got around this by converting it into a numpy array, stacking 3 ontop of each other to give a tensor of shape (224, 224, 3) and then convert back to a PIL.Image object.

Note transforms.Resize((224, 224)) or transforms.ToTensor() is swapping the order of the Tensor to (3, 224, 224), placing the colour channel in the correct spot for pytorch models. (this is does implicitly rather than me permuting the dimensions)


Improvements: 
- I should balance the number of "yes" and "no" examples between the train and test data

- I could use further data augmentation to generate multiple test images from a single train example