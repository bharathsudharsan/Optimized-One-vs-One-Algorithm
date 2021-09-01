# Training up to 50 Class ML Models on Arduino MCUs and Real-time Inference 

In this repo, we provide the code of Opt-OVO, which is an optimized (resource-friendly) version of the popular One-vs-One algorithm which enables high-performance multi-class ML classifier training and inference directly on microcontroller units (MCUs). We evaluate Opt-OVO by performing live ML model training on 4 popular MCU boards using datasets of varying class counts, sizes and feature dimensions.  

**Exciting Finding:** On the  3 $ ESP32, Opt-OVO trained a multi-class ML classifier using a dataset of class count 50 and performed unit inference in super real-time of 6.2 ms.

**Demo:** Video recording in progress

## Table of contents

  * [Opt-OVO Algorithm](#opt-ovo-algorithm)
  * [Datasets, MCU Boards for Training and Inference on MCUs](#datasets--mcu-boards-for-training-and-inference-on-mcus)
    + [Datasets](#datasets)
    + [MCU Boards](#mcu-boards)
  * [Opt-OVO Performance Evaluation](#opt-ovo-performance-evaluation)
    + [Procedure](#procedure)
    + [Training Results - Training Set Size vs Training Time](#training-results---training-set-size-vs-training-time)
    + [Inference Results - Class Size vs Inference Time](#inference-results---class-size-vs-inference-time)
    + [Accuracy of MCUs Trained Models](#accuracy-of-mcus-trained-models)
  * [Extras](#extras)

## Opt-OVO Algorithm

Currently, trainable algorithms are attached to an existing model deployed on MCUs to perform online/continuous learning (for e.g., [TinyTL](https://arxiv.org/abs/2007.11622), [TinyOL](https://arxiv.org/abs/2103.08295)). The training of a full multi-class ML classifier on commodity MCUs, using any existing algorithms is currently not feasible. When analyzing the OVO method, we discovered that the OVO's k(k-1)/2 base learners/classifiers, for a few datasets, contain classifiers that lack significant contributions to the overall multi-class classification result - this occurs when a classifier is already within a big interdependent group. Hence in *Opt-OVO*, we propose to identify and remove the less important base classifiers to improve the resource-friendliness of OVO. 

![alt text](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Opt-OVO-Algorithm.png)

**Opt-OVO 4-steps brief explanation.** In **Step1**, the k(k-1)/2 base classifiers b_i belonging to B are trained with the unseen/fresh local data stream using base learner of choice like SVM, LDA, followed by evaluation of all thus trained base classifiers. Here, each base classifiers b_i produces a binary output ∈ {-1, +1} for each input vector x^(n). In **Step2**, for all test data, we store outcomes of base learners R_i in R_B. Then, we create a correlation matrix C_m using the output of base classifiers stored in R_B. From C_m, we find Corr_{class}, which is the group of highly correlated base classifiers. In **Step3**, from the groups of this found correlated base learners, we create a Probability Table (PT) of each group to know the joint probability of the outcome R_B. These PTs provide the joint probabilities of the outcomes R_B and the groups of correlated classifiers b_{corr} ⊂ Corr_{class} when evaluating using new/unseen data. In **Step4**, finally, we classify for any new multi-class input x^(n) by using thus produced Corr_{class} and set of base classifiers B.

The *Opt-OVO* algorithm detailed explanation available in [Opt-OVO-Algorithm.pdf](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Opt-OVO-Algorithm.pdf

## Datasets, MCU Boards for Training and Inference on MCUs

### Datasets

We converted the listed datasets into MCU readable *.h* files and placed them inside the [Opt-OVO folder](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/tree/main/Opt-OVO), which are used for training multi-class classifiers on MCUs using *Opt-OVO*. The users have to uncomment their dataset of choice (header file at the beginning of the main algorithm program) to use it when training and inference on MCUs.

1. [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/) (64 features, 10 classes, 1797 samples): Data for each digits 0 to 9 is a class. The onboard *Opt-OVO* trained multi-class classifier should distinguish digits, based on the input features.
2. [Australian Sign Language signs](https://archive.ics.uci.edu/ml/datasets/Australian+Sign+Language+signs+(High+Quality)) (22 features, 95 classes, 6650 samples): Here, the class count is 50 since we extracted the files that contain data of 50 Auslan signs varying from *alive* to *more*. Then using *Opt-OVO*, we trained classifiers on MCUs, that distinguish Auslan signs based on the input features.

### MCU Boards

Using Arduino IDE, we upload the *Opt-OVO* algorithm along with the selected/uncommented dataset on the following popular boards. After successful upload, we trained various ML classifier models on MCUs, performed the onboard model evaluation and inference performance evaluation of the thus trained MCU models.

1. B1 [Generic ESP32](https://www.espressif.com/en/products/devkits): Xtensa LX6 @240MHz, 4MB Flash, 520KB SRAM.
2. B2 [ATSAMD21G18 Adafruit METRO](https://www.adafruit.com/product/3505): ARM Cortex-M0+ @48 MHz, 256kB Flash, 32KB SRAM. 
3. B3 [STM32f103c8 Blue Pill](https://stm32-base.org/boards/STM32F103C8T6-Blue-Pill.html): ARM Cortex-M0 @72MHz, 128KB Flash, 20KB SRAM.
4. B4 [nRF52840 Adafruit Feather](https://www.adafruit.com/product/4062): ARM Cortex-M4 @64MHz, 1MB Flash, 256KB SRAM.

## Opt-OVO Performance Evaluation

### Procedure

For evaluation, we selected two multi-class datasets using which the Opt-OVO algorithm trains multi-class classifiers on B1-B4. For the first evaluation round, we use the 64 features Handwritten Digits dataset. Here, we built 3 train sets of various class counts and sizes. For the first train set, we extract data fields corresponding to the handwritten digits 0 to 2 to build a 3 class train set of size 432. The second train set is of class count 5 (digits 0 to 4) and size 720. The last train set of size 1476 contains 10 classes (digits 0 to 9). In all the 3 train sets, each class is of the size 144. 

The second round of evaluation was performed using the 22 features Australian Sign Language signs dataset. Here, we built 8 train sets of different class counts and sizes. For the first train set, we extract data fields corresponding to the **alive, all** and **answer** Auslan signs. Hence, the first set is of class count 3 and size 75. The last set is of class count 50 and size 1250 since it contains data of 50 Auslan signs varying from **alive** to **more**. The in-between train sets contain class counts ranging from 3 to 50, with their corresponding train set size ranging from 0 to 1250. In all the 8 train sets, each class is of the size 25.

We uploaded the *Opt-OVO* algorithm's C implementation on all boards. We then power on each board, connect them to a PC via the serial port to feed the training data, receive training time and classification accuracy from MCUs. The first 70% of data was used for training, the remaining 30% data for evaluation. When we instruct the board to train, *Opt-OVO* iteratively loads the data chunks and trains. Next, we load the test set, infer using the trained models to evaluate the MCU-trained classifiers. 

### Training Results - Training Set Size vs Training Time


![alt text](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Train_time_results.png)

The following analysis is made from the above Figure:

1. Even on the slowest B2 (Adafruit Metro), the Opt-OVO was able to train using a 10 class, 1476 size, 64 dimension Digits dataset in 29.6 sec and could train in 7.6 sec using the 15 class, 375 size, 22 features Australian Sign dataset. 

2. The fastest B1 (ESP32) trained in 0.4 sec for Digits and in 4.7 sec using the 50 class, 1250 size Sign dataset. 

3. In Fig. a, at the individual MCU board level, we show how the training time varies when the class count and train set size increase. 

**AIoT boards:** Using Opt-OVO, users can increase the class count beyond 50 and train without stability issues when they use the emerging AIoT boards like Sipeed MAIX Bit, M5 StickV, Sipeed Maix Amigo that have inbuilt FPU, KPU, FFT hardware capabilities.

### Inference Results - Class Size vs Inference Time


![alt text](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Infer_time_results.png)

To analyze the impact of increasing class count on inference time, in above Fig (left), we feed the Opt-OVO trained models a multi-class data sample (size one) with class count ranging from 0 to 10 and from 0 to 50 in above Fig (right). For statistical validation, the plotted inference time corresponds to the average of 5 runs. The following analysis is made from the above Fig:

1. Even for the high dimensional Digits dataset, our method achieves real-time unit inference, 11.8 ms, even on the slowest B2. 

2. The fastest B1 was able to infer for a 50 class input in 6.2 ms. 

3. In Fig (right), when the class count increased from 15 to 25, the inference time increased by 0.9 ms, from 25 to 40 by 2.4ms and 40 to 50 by 2.2 ms, showing an almost linear relationship. 

4. Overall, it is apparent that the Opt-OVO trained classifiers perform onboard unit inference for multi-class data in super real-time, within a second, across various MCUs.

### Accuracy of MCUs Trained Models

The accuracy of the Opt-OVO trained models on MCUs with the train samples used are provided in [Train_time_and_accuracy_results.xlsx](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Train_time_and_accuracy_results.xlsx). We are not presenting the explicit performance comparing of the classifiers trained using Opt-OVO, with the classifiers trained on high resource setups using *Python scikit-learn* since we achieve similar accuracies when experimenting using the same setup and datasets. 

## Extras

**Additional Datasets:** In [TinyML datasets](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/tree/main/TinyML%20Datasets) folder, the following 7 datasets are made available as *.h* files that can be used for training and inference using *Opt-OVO* on MCU boards. The details in brackets are samples size x features count x classes count.
1. [EMG](https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures) (1648 x 63 x 5 ): Raw EMG data recorded by MYO Thalmic bracelet worn on a users forearm. This bracelet is equipped with eight sensors equally spaced around the forearm that simultaneously acquire myographic signals.
2. [Gas Sensor Array Drift](https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset) (1000 x 128 x 6): Contains measurements from 16 chemical sensors utilized in simulations for drift compensation in a discrimination task of 6 gases at various levels of concentrations.
3. [Gesture Phase Segmentaion](https://archive.ics.uci.edu/ml/datasets/gesture+phase+segmentation) (1000 x 19 x 5):  Contains features extracted from 7 videos with people gesticulating, aiming at studying Gesture Phase Segmentation. It contains 50 attributes divided into two files for each video.
4. [Human Activity](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) (10299 x 561 x 6): Database built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors.
5. [Mammographic Mass](http://archive.ics.uci.edu/ml/datasets/mammographic+mass) (830 x 4 x 2): Dataset for discrimination of benign and malignant mammographic masses based on BI-RADS attributes and the patient's age.
6. [Sensorless Drive Diagnosis](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis)  (1000 x 48 x 11): Features are extracted from motor current. The motor has intact and defective components. This results in 11 different classes with different conditions.
7. [Sport Activity](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities) (4800 x 180 x 10): The dataset comprises motion sensor data of 19 daily and sports activities each performed by 8 subjects in their own style for 5 minutes. Five Xsens MTx units are used on the torso, arms, and legs.
 