# Face Recognition using MATLAB and AlexNet

## Introduction
This project implements a real-time face recognition system using MATLAB and AlexNet, a deep convolutional neural network (CNN). The system captures face images, trains a model using AlexNet, and classifies detected faces in real-time.

## Features
- **Real-time Face Detection**: Uses a webcam to detect faces.
- **Face Data Collection**: Captures and saves face images.
- **Deep Learning Model**: Trains an AlexNet-based classifier.
- **Live Face Recognition**: Classifies detected faces in real-time.

## Prerequisites
Ensure you have the following installed:
- MATLAB (with Deep Learning Toolbox)
- Webcam support
- AlexNet pre-trained model
- Vision Toolbox for face detection

## Installation
1. Install MATLAB and required toolboxes.
2. Connect a webcam to your computer.
3. Ensure the dataset for training is stored properly.

## Usage
### 1. Data Collection
Run the following script to collect face images:
```matlab
clc;
clear all;
close all;
warning off;
cao=webcam;
faceDetector=vision.CascadeObjectDetector;
c=150;
temp=0;
while true
    e=cao.snapshot;
    bboxes = step(faceDetector, e);
    if (sum(sum(bboxes)) ~= 0)
        if (temp >= c)
            break;
        else
            es = imcrop(e, bboxes(1,:));
            es = imresize(es, [227 227]);
            filename = strcat(num2str(temp), '.bmp');
            imwrite(es, filename);
            temp = temp + 1;
            imshow(es);
            drawnow;
        end
    else
        imshow(e);
        drawnow;
    end
end
```

### 2. Training the Model
Run the following script to train a deep learning model:
```matlab
clc;
clear all;
close all;
warning off;
g = alexnet;
layers = g.Layers;
layers(23) = fullyConnectedLayer(2);
layers(25) = classificationLayer;
allImages = imageDatastore('datasets for face recognition', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 20, 'MiniBatchSize', 64);
myNet1 = trainNetwork(allImages, layers, opts);
save myNet1;
```

### 3. Face Recognition
Run the script below for real-time face recognition:
```matlab
clc;
close all;
clear;
c = webcam;
load myNet1;
faceDetector = vision.CascadeObjectDetector;
while true
    e = c.snapshot;
    bboxes = step(faceDetector, e);
    if (sum(sum(bboxes)) ~= 0)
        es = imcrop(e, bboxes(1,:));
        es = imresize(es, [227 227]);
        label = classify(myNet1, es);
        image(e);
        title(char(label));
        drawnow;
    else
        image(e);
        title('No Face Detected');
    end
end
```

## Project Structure
```
|-- datasets for face recognition/
|   |-- person1/
|   |   |-- img1.bmp
|   |   |-- img2.bmp
|   |-- person2/
|-- face_recognition.m
|-- train_model.m
|-- test_model.m
|-- myNet1.mat
```

## Notes
- The dataset must be organized with separate folders for each person.
- Adjust `MaxEpochs` and `MiniBatchSize` for better performance.
- Make sure the `alexnet` model is available in MATLAB.

## License
This project is open-source and free to use for educational purposes.

## Author
[Awais Raza](https://www.linkedin.com/in/awais-raza-886533283/)

