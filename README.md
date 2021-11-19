# Source code (Pytorch) for the paper: "A machine and human reader study on AI diagnosis model safety under attacks of adversarial images"
# Contact: zhouqianweischolar@gmail.com (Qianwei Zhou) and wus3@upmc.edu (Shandong Wu)
# Configure the Enviroment
## Depending on speed of your Internet connection，installation may take hours.
## In our implemenation, we used python 3.7.4, Ubuntu 18.04.5 LTS with GPU.    
1. nvidia gpu driver version: 455.28
2. cuda version: 11.1
3. GPU memory >= 24GB x 2 or 48GB x 1 
4. install miniconda
5. `$ conda create --name testENV --file package-list.txt -c pytorch`
6. `$ pip install pypng`

## If without GPU, install on Ubuntu 20.04.1 LTS.
1. install miniconda 
2. RAM >= 48GB
3. `$ conda create --name testENV --file package-list-cpu.txt cpuonly -c pytorch`
4. `$ pip install pypng`

# You can train/test the AI-CAD classification models and GAN models by following instructions below.
* Imaging data are not available, but should be placed in \Samples\realImages and \Samples\fakeImages.  

# Prepare Imaging Data
1. Adjust original mammgorma images to the resolutions reported in the paper.
2. Place traing data to folder /Samples/realImages/, and in .txt files listing the images (PNG format).
    1. In cancerList.txt, list all images of positive cases.
    2. In cancerList-train.txt, cancerList-valid.txt, and cancerList-test.txt, list positive images for training, validation, and testing respectively.
    3. In negb9List.txt, list all images of negative/normal cases.
    4. In negb9List-train.txt, negb9List-valid.txt, and negb9List-test.txt, list negative/normal images for training, validation, and testing respectively.
    5. Note, the code requires a PNG file name starting with a unique global ID number (e.g., 1049L-CC-pos.png).  

# Train AI-CAD Classifier
`$ python trainClas1728pal134.py`  
* When no GPU is avaiable, the script may take 37GB RAM and 30 mins of 8 Intel Xeon E5-2620 cores for one epoch.  
* Output: 
    * The AI-CAD models will be in the folder /checkfolder-Classifier/modelfolder.   
    * They may look like VN\_Ep-0.pth (the model of Vnet), FN\_Ep-0.pth (the model of Fnet without the last fully connected layer), CN\_Ep-0.pth (the last fully connected layer).  
  

# Train GAN Generators
* In files trainV1728negPal41.py and trainV1728posPal41.py, please set preTrainedModel to the classifier learned by trainClas1728pal134.py.
* When no GPU is avaiable, the scripts may take 25GB RAM and 15 mins of 8 Intel Xeon E5-2620 cores for one epoch.  

`$ python trainV1728negPal41.py` to train negative-looking fake/adversarial image generator.  
* Output:
    * The generator and discriminator models will be in the folder /checkfolder-negativeLookingGenerator/modelfolder.
    * They may look like VN\_Ep-0.pth (the model of Vnet), FN\_Ep-0.pth (the model of Dnet without the last fully connected layer), DN\_Ep-0.pth (the last fully connected layer), GN\_Ep-0.pth (the model of Gnet).

`$ python trainV1728posPal41.py` to train positive-looking fake/adversarial image generator.    
* Output:
    * The generator and discriminator models will be in the folder /checkfolder-positiveLookingGenerator/modelfolder.
    * They may look like VN\_Ep-0.pth (the model of Vnet), FN\_Ep-0.pth (the model of Dnet without the last fully connected layer), DN\_Ep-0.pth (the last fully connected layer), GN\_Ep-0.pth (the model of Gnet).


# Test the AI-CAD Classifier on Real Images  
* In file trainClas1728pal134-score-testing.py, please set AUCepoch as the epoch number (100 for example) of the models that you are going to test. 
* Please copy the target models to /Samples/Classifier, for example, VN\_Ep-100.pth, FN\_Ep-100.pth, CN\_Ep-100.pth.

`$ python trainClas1728pal134-score-testing.py`  

* Output: the code will predict the probability of positive and negative of the real images. The predicted probabilities will be listed in the file /Samples/testDataScores-100/Scores-test-real.txt.  
    * Take images 19453 and 3917 as exmpales to explain the output in file Scores-test-real.txt.  
    * Image 19453 has label 0 which means it's a cancer case. The classifier predicts that this image has probability of 0.82 to be positive.  
    * Image 3917 has label 1 which means it's a negative case. The classifier predicts that this image has probability of 0.73 to be negative.  
    * This indicates the classifer correctly predicted the two real images. All the output in Scores-test-real.txt can be exlained this way similarly.
* When no GPU is avaiable, the script may take 7GB RAM and 5 mins of 8 Intel Xeon E5-2620 cores.


# Test the Positive-Looking Fake/Adversarial Image Generator  
* In the file testV1728posPal41.py, please set epTEST as the epoch number (100 for example) of the models that you are going to test.  
* Please copy the target models (for example, VN\_Ep-100.pth, FN\_Ep-100.pth, DN\_Ep-100.pth, GN\_Ep-100.pth) to folder /Samples/posGAN.  

`$ python testV1728posPal41.py`  

* Output: the code will generate positive-looking fake/adversarial images from negative images.   
    * Generated fake images will be in the folder /Samples/fakeImages/testPositiveEP100
    * For example: /Samples/realImages/3917L-CC-neg.png ---> /Samples/fakeImages/testPositiveEP100/3917fakePos.png  
    * You may need to make a /Samples/fakeImages/cancerList.txt to list all the positive-looking fake images.
* When no GPU is avaiable, the script may take 7GB RAM and 10 mins of 8 Intel Xeon E5-2620 cores.

# Test the Negative-Looking Fake/Adversarial Image Generator  
* In the file testV1728negPal41.py, please set epTEST as the epoch number (100 for example) of the models that you are going to test.  
* Please copy the target models (for example, VN\_Ep-100.pth, FN\_Ep-100.pth, DN\_Ep-100.pth, GN\_Ep-100.pth) to folder /Samples/negGAN. 

`$ python testV1728negPal41.py`  

* Output: the code will generate negative-looking fake/adversarial images from positive images.
    * generated fake images will be in the folder /Samples/fakeImages/testNegativeEP100.
    * For example: /Samples/realImages/19453L-CC-pos.png ---> /Samples/fakeImages/testNegativeEP100/19453fakeNeg.png   
    * You may need to make a /Samples/fakeImages/negb9List.txt to list all the negative-looking fake images.  
* When no GPU is avaiable, the script may take 7GB RAM and 10 mins of 8 Intel Xeon E5-2620 cores.

# Test the Classifier on Fake/Adversarial Images  
* In file trainClas1728pal134-score-testing-fake.py, please set AUCepoch as the epoch number (100 for example) of the models that you are going to test. 
* Please copy the target models (for example, VN\_Ep-100.pth, FN\_Ep-100.pth, CN\_Ep-100.pth) to folder /Samples/Classifier.

`$ python trainClas1728pal134-score-testing-fake.py`  

* Output: the code will predict the probability of positive and negative of the fake images. The probabilities will be listed in the file /Samples/testFakeDataScores-100/Scores-test-fake.txt.  
    * Take images 19453 and 3917 as exmpales to explain the output in file Scores-test-fake.txt.  
    * The positive-looking fake/adversarial image 3917 has label 0 which means its label is positive, while the label of the orignal image 3917 is negative. The classifier predicts that this image has probability of 0.80 to be positive.   
    * The negative-looking fake/adversarial image 19453 has label 1 which means its label is negative, while the label of the orignal image 19453 is positive. The classifier predicts that this image has probability of 0.84 to be negative.  
    * This indicates that the 2 adversarial samples are successful to attack the AI-CAD model because they enabled the classifer to output an opposite diagnosis. All the output in Scores-test-fake.txt can be exlained this way similarly.
* When no GPU is avaiable, the script may take 7GB RAM and 10 mins of 8 Intel Xeon E5-2620 cores.  

# The Work is Licensed with Apache License Version 2.0.
