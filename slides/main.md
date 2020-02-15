name: inverse
class: center, middle, inverse
layout: true
title: Python-basic

---
class: titlepage, no-number

# Machine Learning with MATLAB
## .gray.author[Youngjae Yu]

### .x-small[https://github.com/yj-yu/matlab-ml]
### .x-small[https://yj-yu.github.io/matlab-ml]

.bottom.img-66[ ![](images/snu-logo.png) ]

---
layout: false

## About

- MATLAB Deep Learning Toolbox Tutorial
- Deep Network Designer : Classify and Finetune GoogLeNet 
- Image Classification
- Transfer Learning
- Deeplearning Feature Extraction

---

template: inverse

# MATLAB Deep Learning Toolbox

Deep Learning Toolbox™는 심층 신경망의 계층을 만들고 상호 연결하는 간단한 MATLAB® 명령을 제공합니다. 

다양한 예제와 사전 훈련된 네트워크가 제공되기 때문에 고급 컴퓨터 비전 알고리즘이나 신경망에 대한 사전 지식이 없어도 쉽게 MATLAB을 심층 학습에 사용할 수 있습니다.


---

## Use Computer Vision for Your Research

Flowchart of the proposed convolutional neural network (CNN) based classification model.

.center.img-66[ ![](images/app1.png) ]


---

## Use Computer Vision for Your Research

Rock Detection in a Mars-Like Environment Using a CNN

.center.img-40[ ![](images/app2.png) ]


---

## Use Computer Vision for Your Research

터널막장 이미지로부터 암반등급을 5개로 분류

.center.img-80[ ![](images/app3.jpg) ]

---

## Use Computer Vision for Your Research

Detect cracks captured in videos of nuclear reactors. 

.center.img-50[ ![](images/app6.jpg) ]


Tunnel Crack Detection with Deep Learning

.center.img-50[ ![](images/app8.png) ]


---



## Deel Learning Classifier

- Deep learning uses neural networks to learn useful representations of features directly from data. 

- Neural networks combine multiple nonlinear processing layers, using simple elements operating in parallel and inspired by biological nervous systems. 

- Deep learning models can achieve state-of-the-art accuracy in object classification, sometimes exceeding human-level performance.

.center.img[ ![](images/cnn.png) ]


---

## Conv Network

.center.img[ ![](images/cnn.png) ]

MATLAB을 이용해 컨벌루션 신경망의 계층을 모델링 할 수 있습니다.

```python
layers = [imageInputLayer([224 224 1])
          convolution2dLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(100)
          softmaxLayer
          classificationLayer];
```

---

## Conv Network

trainingOptions 함수를 사용하여 훈련 옵션을 지정
```python
options = trainingOptions('sgdm');
```

그런 다음 trainNetwork 함수를 사용하여 훈련 데이터로 네트워크를 훈련시킬 수 있습니다. 
```python
convnet = trainNetwork(data,layers,options);
```

---

## Conv Network


.center.img[ ![](images/cnn2.png) ]


---


template: inverse

# Simple Deep Learning Training


---

## Create Simple Deep Learning Network for Classification

Let's train a simple convolutional neural network for deep learning classification. 

- Load and explore image data.

- Define the network architecture.

- Specify training options.

- Train the network.

- Predict the labels of new data and calculate the classification accuracy.

---

## Load and Explore Image Data

Load the digit sample data as an image datastore. 
- imageDatastore automatically labels the images based on folder names and stores the data as an ImageDatastore object. 
- An image datastore enables you to store large image data, including data that does not fit in memory, and efficiently read batches of images during training of a convolutional neural network.

```python
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
```

---
Display some of the images in the datastore.

```python
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
```

.center.img-66[ ![](images/digits.png) ]


---

## Load and Explore Image Data

Calculate the number of images in each category. 

- labelCount is a table that contains the labels and the number of images having each label. 
- The datastore contains 1000 images for each of the digits 0-9, for a total of 10000 images. 
- You can specify the number of classes in the last fully connected layer of your network as the OutputSize argument.

```python
labelCount = countEachLabel(imds)
```

---

## Load and Explore Image Data

You must specify the size of the images in the input layer of the network. 
- Check the size of the first image in digitData. Each image is 28-by-28-by-1 pixels.

```python
img = readimage(imds,1);
size(img)
```

---

## Specify Training and Validation Sets

Divide the data into training and validation data sets, so that each category in the training set contains 750 images, 
- the validation set contains the remaining images from each label. 
- splitEachLabel splits the datastore digitData into two new datastores, trainDigitData and valDigitData.

```python
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
```

---

## Define Network Architecture

Define the convolutional neural network architecture.

```python
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
```

---

## Define Network Architecture

###Image Input Layer 
- An imageInputLayer is where you specify the image size, which, in this case, is 28-by-28-by-1. 
- These numbers correspond to the height, width, and the channel size. 
- The digit data consists of grayscale images, so the channel size (color channel) is 1. 
- For a color image, the channel size is 3, corresponding to the RGB values. 
- You do not need to shuffle the data because trainNetwork, by default, shuffles the data at the beginning of training. 
- trainNetwork can also automatically shuffle the data at the beginning of every epoch during training.

---

###Convolutional Layer 
- In the convolutional layer, the first argument is filterSize, which is the height and width of the filters the training function uses while scanning along the images. 
- In this example, the number 3 indicates that the filter size is 3-by-3. 
- You can specify different sizes for the height and width of the filter. 
- The second argument is the number of filters, numFilters, which is the number of neurons that connect to the same region of the input. This parameter determines the number of feature maps. 
- Use the 'Padding' name-value pair to add padding to the input feature map. For a convolutional layer with a default stride of 1, 'same' padding ensures that the spatial output size is the same as the input size. 
- You can also define the stride and learning rates for this layer using name-value pair arguments of convolution2dLayer.

---

###Batch Normalization Layer 
- Batch normalization layers normalize the activations and gradients propagating through a network, making network training an easier optimization problem. 
- Use batch normalization layers between convolutional layers and nonlinearities, such as ReLU layers, to speed up network training and reduce the sensitivity to network initialization. 
- Use batchNormalizationLayer to create a batch normalization layer.

---

###ReLU Layer 
- The batch normalization layer is followed by a nonlinear activation function. 
- The most common activation function is the rectified linear unit (ReLU). 
- Use reluLayer to create a ReLU layer.

---

###Max Pooling Layer 
- Convolutional layers (with activation functions) are sometimes followed by a down-sampling operation that reduces the spatial size of the feature map and removes redundant spatial information. 
- Down-sampling makes it possible to increase the number of filters in deeper convolutional layers without increasing the required amount of computation per layer. 
- One way of down-sampling is using a max pooling, which you create using maxPooling2dLayer. 
- The max pooling layer returns the maximum values of rectangular regions of inputs, specified by the first argument, poolSize.
- The 'Stride' name-value pair argument specifies the step size that the training function takes as it scans along the input.

---

###Fully Connected Layer 
- The convolutional and down-sampling layers are followed by one or more fully connected layers. 
- As its name suggests, a fully connected layer is a layer in which the neurons connect to all the neurons in the preceding layer. 
- This layer combines all the features learned by the previous layers across the image to identify the larger patterns. 
- The last fully connected layer combines the features to classify the images. Therefore, the OutputSize parameter in the last fully connected layer is equal to the number of classes in the target data. 
- In this example, the output size is 10, corresponding to the 10 classes. 
- Use fullyConnectedLayer to create a fully connected layer.

---

###Softmax Layer 
- The softmax activation function normalizes the output of the fully connected layer. 
- The output of the softmax layer consists of positive numbers that sum to one, which can then be used as classification probabilities by the classification layer. 
- Create a softmax layer using the softmaxLayer function after the last fully connected layer.

###Classification Layer 
- The final layer is the classification layer. This layer uses the probabilities returned by the softmax activation function for each input to assign the input to one of the mutually exclusive classes and compute the loss. 
- To create a classification layer, use classificationLayer.


---

## Specify Training Options

After defining the network structure, specify the training options. 

```python
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
```

---

## Train Network Using Training Data

Train the network using the architecture defined by layers, the training data, and the training options. 

The training progress plot shows the mini-batch loss and accuracy and the validation loss and accuracy. 
- For more information on the training progress plot, see Monitor Deep Learning Training Progress. 
- The loss is the cross-entropy loss. 
- The accuracy is the percentage of images that the network classifies correctly.

```python
net = trainNetwork(imdsTrain,layers,options);
```

---

## Train Network Using Training Data

.center.img-80[ ![](images/curve.png) ]


---

## Classify Validation Images and Compute Accuracy

Predict the labels of the validation data using the trained network, and calculate the final validation accuracy. 
- Accuracy is the fraction of labels that the network predicts correctly. 
- In this case, more than 99% of the predicted labels match the true labels of the validation set.

```python
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
```

accuracy = 0.9872

---

template: inverse


# 미리 학습된 강력한 네트워크를 사용해 보자.


---

## Load Pretrained Network

how to load GoogLeNet network to classify a new collection of images.

Load Pretrained Network
- Load a pretrained GoogLeNet network. If you need to download the network, use the download link.

```python
net = googlenet;
```

---

## Input

The image that you want to classify must have the same size as the input size of the network. For GoogLeNet, the first element of the Layers property of the network is the image input layer. 
- The network input size is the InputSize property of the image input layer.

```python
inputSize = net.Layers(1).InputSize
```
inputSize = 1×3

   224   224     3


---

## Output

The final element of the Layers property is the classification output layer. 
- The ClassNames property of this layer contains the names of the classes learned by the network. 
- View 10 random class names out of the total of 1000.

```python
classNames = net.Layers(end).ClassNames;
numClasses = numel(classNames);
disp(classNames(randperm(numClasses,10)))
```
    'speedboat'
    'window screen'
    'isopod'
    'wooden spoon'
    'lipstick'
    'drake'
    'hyena'
    'dumbbell'
    'strawberry'
    'custard apple'

---

## Read and Resize Image

-The image that you want to classify must have the same size as the input size of the network. 

- For GoogLeNet, the network input size is the InputSize property of the image input layer.

- Read the image that you want to classify and resize it to the input size of the network. This resizing slightly changes the aspect ratio of the image.

```python
I = imread("peppers.png");
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));
```


---

## Classify and Display Image

Classify and display the image with the predicted label.

```python
[label,scores] = classify(net,I);
figure
imshow(I)
title(string(label))
```

.center.img-66[ ![](images/img0.png) ]



---

## Display Top Predictions

Display the top five predicted labels and their associated probabilities as a histogram. 

- Because the network classifies images into so many object categories, and many categories are similar, it is common to consider the top-five accuracy when evaluating networks. 

- The network classifies the image as a bell pepper with a high probability.

```python

[~,idx] = sort(scores,'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)
```

---

## Display Top Predictions

.center.img-66[ ![](images/img7.png) ]



---


template: inverse

# Transfer Learning

Tuning pretrained network to your own purpose


---

## Start from Transfer Learning

You can take a pretrained network and use it as a starting point to learn a new task.

- Fine-tuning a network with transfer learning is much faster and easier than training from scratch.

- You can quickly make the network learn a new task using a smaller number of training images. 

- The advantage of transfer learning is that the pretrained network has already learned a rich set of features that can be applied to a wide range of other similar tasks.

---

## Start from Transfer Learning

- For example, if you take a network trained on thousands or millions of images, you can retrain it for new object detection using only hundreds of images. 

- You can effectively fine-tune a pretrained network with much smaller data sets than the original training data. If you have a very large dataset, then transfer learning might not be faster than training a new network.

.center.img[ ![](images/img8.png) ]


---


## Transfer learning enables you to

- Transfer the learned features of a pretrained network to a new problem

- Transfer learning is faster and easier than training a new network

- Reduce training time and dataset size

- Perform deep learning without needing to learn how to create a whole new network


---

## Finetune Pretrained Network

how to finetune GoogLeNet network to classify a new collection of images.

Load Pretrained Network

```python
net = googlenet;
```

---

## Finetune Pretrained Network

Use functions such as googlenet to get links to download pretrained networks from the Add-On Explorer. 
- The following table lists the available pretrained networks trained on ImageNet and some of their properties. 
- The network depth is defined as the largest number of sequential convolutional or fully connected layers on a path from the input layer to the output layer. 
- The inputs to all networks are RGB images.

.center.img[ ![](images/pretrain.png) ]

---

## Finetune Pretrained Network

If you need to classify scene

use googlenet('Weights','places365')

'places365' is outdoor scene dataset

---

## Finetune Pretrained Network

### Tip !

Faster Network : SqueezeNet or GoogLeNet (low parameters)

Performance : Inception-v3 or a ResNet (high parameters)

.center.img-80[ ![](images/pretrained2.png) ]


---

## Import Network into Deep Network Designer

Open Deep Network Designer.

```bash
deepNetworkDesigner
```

Click Import and select the network from the workspace. 
- Deep Network Designer displays a zoomed out view of the whole network. 
- Explore the network plot. 
- To zoom in with the mouse, use Ctrl+scroll wheel.


---

## Import Network into Deep Network Designer

.center.img-66[ ![](images/img1.png) ]


---


## Edit Network for Transfer Learning

- To retrain a pretrained network to classify new images, replace the final layers with new layers adapted to the new data set. You must change the number of classes to match your data.

- Drag a new fullyConnectedLayer from the Layer Library onto the canvas. 

- Edit the OutputSize to the number of classes in the new data, in this example, 5.


.center.img-33[ ![](images/img2.png) ]


---


## Edit Network for Transfer Learning


- Edit learning rates to learn faster in the new layers than in the transferred layers. 

- Set WeightLearnRateFactor and BiasLearnRateFactor to 10. 

- Delete the last fully connected and connect up your new layer instead.

.center.img-33[ ![](images/img2.png) ]


---


## Edit Network for Transfer Learning

- Replace the output layer. 

- Scroll to the end of the Layer Library and drag a new classificationLayer onto the canvas. 

- Delete the original output layer and connect up your new layer instead.

.center.img-33[ ![](images/img3.png) ]




---

## Check Network

To make sure your edited network is ready for training, click Analyze, and ensure the Deep Learning Network Analyzer reports zero errors.

.center.img[ ![](images/img4.png) ]

---

## Export Network for Training

- Return to the Deep Network Designer and click Export. 

- Deep Network Designer exports the network to a new variable called lgraph_1 containing the edited network layers. 

- You can now supply the layer variable to the trainNetwork function. 

- You can also generate MATLAB® code that recreates the network architecture and returns it as a layerGraph object or a Layer array in the MATLAB workspace.

---

## Load Data and Train Network

- Unzip and load the new images as an image datastore. 

- Divide the data into 70% training data and 30% validation data.


```python
unzip('MerchData.zip');
imds = imageDatastore('MerchData','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
```
- Resize images to match the pretrained network input size.

```python
augimdsTrain = augmentedImageDatastore([224 224],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224],imdsValidation);
```


---

- Specify training options.
```python
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',6, ...
    'Verbose',false, ...
    'Plots','training-progress');
```

- Specify the mini-batch size, that is, how many images to use in each iteration.

- Specify a small number of epochs. An epoch is a full training cycle on the entire training data set. For transfer learning, you do not need to train for as many epochs. Shuffle the data every epoch.

- Set InitialLearnRate to a small value to slow down learning in the transferred layers.

- Specify validation data and a small validation frequency.

- Turn on the training plot to monitor progress while you train.


---

## Search your target project 

To train the network, supply the layers exported from the app, lgraph_1, the training images, and options, to the trainNetwork function. 
By default, trainNetwork uses a GPU if available (requires Parallel Computing Toolbox™). 
Otherwise, it uses a CPU. Training is fast because the data set is so small.

```python
netTransfer = trainNetwork(augimdsTrain,lgraph_1,options);
```

.center.img-66[ ![](images/img5.png) ]


---

## Test Trained Network


Classify the validation images using the fine-tuned network, and calculate the classification accuracy.

```python
[YPred,probs] = classify(netTransfer,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
accuracy = 1
```

---

## Test Trained Network

Display four sample validation images with predicted labels and predicted probabilities.
```python

idx = randperm(numel(augimdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
```

---

## Test Trained Network


.center.img-66[ ![](images/img6.png) ]


---

## More details

To learn more and try other pretrained networks, see Deep Network Designer official page.

https://kr.mathworks.com/help/deeplearning/ref/deepnetworkdesigner-app.html

---

template: inverse

# Try Deep Learning in 10 Lines of MATLAB Code


---

## WebCAM Demo

This example shows how to use deep learning to identify objects on a live webcam using only 10 lines of MATLAB code. 

- Try the example to see how simple it is to get started with deep learning in MATLAB.

- Run these commands to get the downloads if needed, connect to the webcam, and get a pretrained neural network.
```python
camera = webcam; % Connect to the camera
net = alexnet;   % Load the neural network
```

---

## WebCAM Demo

If you need to install the webcam and alexnet add-ons, a message from each function appears with a link to help you download the free add-ons using Add-On Explorer. 
- Alternatively, see Deep Learning Toolbox Model for AlexNet Network and MATLAB Support Package for USB Webcams.

After you install Deep Learning Toolbox Model for AlexNet Network, you can use it to classify images. 
- AlexNet is a pretrained convolutional neural network (CNN) that has been trained on more than a million images and can classify images into 1000 object categories (for example, keyboard, mouse, coffee mug, pencil, and many animals).

---

## WebCAM Demo

Run the following code to show and classify live images. 
- Point the webcam at an object and the neural network reports what class of object it thinks the webcam is showing. 
- It will keep classifying images until you press Ctrl+C. The code resizes the image for the network using imresize.

```python
while true
    im = snapshot(camera);       % Take a picture
    image(im);                   % Show the picture
    im = imresize(im,[227 227]); % Resize the picture for alexnet
    label = classify(net,im);    % Classify the picture
    title(char(label));          % Show the class label
    drawnow
end
```

.center.img-20[ ![](images/cup.png) ]

---

template: inverse

# Deeplearning Feature Extraction

For very small datasets? (fewer than about 20 images per class)

---

## Feature Extraction tip

Feature extraction is an easy and fast way to use the power of deep learning without investing time and effort into training a full network. 
- it is especially useful if you do not have a GPU. 
- You extract learned image features using a pretrained network, and then use those features to train a classifier, such as a support vector machine using fitcsvm (SVM).

Try feature extraction when your new data set is very small. 
- Since you only train a simple classifier on the extracted features, training is fast. 
- It is also unlikely that fine-tuning deeper layers of the network improves the accuracy since there is little data to learn from.


---

## Feature Extraction tip

If your data is very similar to the original data, 
- then the more specific features extracted deeper in the network are likely to be useful for the new task.

If your data is very different from the original data, 
- then the features extracted deeper in the network might be less useful for your task. 
- Try training the final classifier on more general features extracted from an earlier network layer. 

If the new data set is large, 
- then you can also try training a network from scratch.

---


## Extract Image Features Using Pretrained Network

This slide shows 
- how to extract learned image features from a pretrained convolutional neural network, 
- and use those features to train an image classifier. 

Feature extraction is the easiest and fastest way to use the representational power of pretrained deep networks. 
- For example, you can train a support vector machine (SVM) using fitcecoc (Statistics and Machine Learning Toolbox™) on the extracted features. 
- it is a good starting point if you do not have a GPU to accelerate network training with.


---

## Load Data

Unzip and load the sample images as an image datastore. 
- imageDatastore automatically labels the images based on folder names and stores the data as an ImageDatastore object. 
- An image datastore lets you store large image data, including data that does not fit in memory. 
- Split the data into 70% training and 30% test data.

```python
unzip('MerchData.zip');
imds = imageDatastore('MerchData','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
```

---

## Load Data

There are now 55 training images and 20 validation images in this very small data set. Display some sample images.

```python
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
```

---

## Load Data

.center.img-66[ ![](images/img9.png) ]

---

## Load Pretrained Network

Load a pretrained ResNet-18 network. 

If the Deep Learning Toolbox Model for ResNet-18 Network support package is not installed, then the software provides a download link. 
- ResNet-18 is trained on more than a million images and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. 
- As a result, the model has learned rich feature representations for a wide range of images.

```python
net = resnet18
```

---

## Load Pretrained Network

Analyze the network architecture. T

he first layer, the image input layer, requires input images of size 224-by-224-by-3, where 3 is the number of color channels.

```python
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net)
```

---

## analyzeNetwork

.center.img[ ![](images/img10.png) ]


---

## Extract Image Features

The network requires input images of size 224-by-224-by-3, 

- but the images in the image datastores have different sizes. 

- To automatically resize the training and test images before they are input to the network, create augmented image datastores, specify the desired image size, and use these datastores as input arguments to activations.

```python
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
```

---

## Extract Image Features

The network constructs a hierarchical representation of input images. Deeper layers contain higher-level features, constructed using the lower-level features of earlier layers. 

- To get the feature representations of the training and test images, use activations on the the global pooling layer, 'pool5', at the end of the network. 

- The global pooling layer pools the input features over all spatial locations, giving 512 features in total.

```python
layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

whos featuresTrain
```

---

## Extract Image Features

Extract the class labels from the training and test data.

```python
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
```

---

## Fit Image Classifier

Use the features extracted from the training images as predictor variables and fit a multiclass support vector machine (SVM) using fitcecoc (Statistics and Machine Learning Toolbox).

```python
classifier = fitcecoc(featuresTrain,YTrain);
```

---

## Classify Test Images

Classify the test images using the trained SVM model using the features extracted from the test images.

```python
YPred = predict(classifier,featuresTest);
```

Display four sample test images with their predicted labels.

```python
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end
```

---

## Classify Test Images

.center.img-66[ ![](images/img11.png) ]

---

## Classify Test Images

Calculate the classification accuracy on the test set. Accuracy is the fraction of labels that the network predicts correctly.

```python
accuracy = mean(YPred == YTest)
```

accuracy = 1

---

## Train Classifier on Shallower Features

You can also extract features from an earlier layer in the network and train a classifier on those features. 

- Earlier layers typically extract fewer, shallower features, have higher spatial resolution, and a larger total number of activations. 

- Extract the features from the 'res3b_relu' layer. This is the final layer that outputs 128 features and the activations have a spatial size of 28-by-28.

```python
layer = 'res3b_relu';
featuresTrain = activations(net,augimdsTrain,layer);
featuresTest = activations(net,augimdsTest,layer);
whos featuresTrain
```

---

## Train Classifier on Shallower Features

The extracted features used in the first part of this example were pooled over all spatial locations by the global pooling layer. 

- To achieve the same result when extracting features in earlier layers, manually average the activations over all spatial locations. 

- To get the features on the form N-by-C, where N is the number of observations and C is the number of features, remove the singleton dimensions and transpose.

```python
featuresTrain = squeeze(mean(featuresTrain,[1 2]))';
featuresTest = squeeze(mean(featuresTest,[1 2]))';
whos featuresTrain
```

---

## Train Classifier on Shallower Features

Train an SVM classifier on the shallower features. Calculate the test accuracy.

```python
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == YTest)
```

accuracy = 0.9500


---

## Train Classifier on Shallower Features

Both trained SVMs have high accuracies. 

- If the accuracy is not high enough using feature extraction, then try transfer learning instead. 

- transfer learning ensures higher accuracy, but requires data and resources (GPU, Memory etc)



---
name: last-page
class: center, middle, no-number
## Thank You!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
