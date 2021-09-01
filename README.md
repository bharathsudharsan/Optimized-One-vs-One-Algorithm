# Training up to 50 Class ML Models on Arduino MCUs and Real-time Inference 

Table of contents
=================

<!--ts-->
   * [Installation](#installation)
   * [Usage](#usage)
      * [STDIN](#stdin)
      * [Local files](#local-files)
      * [Remote files](#remote-files)
      * [Multiple files](#multiple-files)
      * [Combo](#combo)
      * [Auto insert and update TOC](#auto-insert-and-update-toc)
      * [GitHub token](#github-token)
      * [TOC generation with Github Actions](#toc-generation-with-github-actions)
   * [Tests](#tests)
   * [Dependency](#dependency)
   * [Docker](#docker)
     * [Local](#local)
     * [Public](#public)
<!--te-->

In this repo, we provide the code of Opt-OVO, which is an optimized (resource-friendly) version of the popular One-vs-One algorithm which enables high-performance multi-class ML classifier training and inference directly on microcontroller units (MCUs). We evaluate Opt-OVO by performing live ML model training on 4 popular MCU boards using datasets of varying class counts, sizes and feature dimensions.  

**Exciting finding** On the  3 $ ESP32, Opt-OVO trained a multi-class ML classifier using a dataset of class count 50 and performed unit inference in super real-time of 6.2 ms.

## Opt-OVO Algorithm Design



## Datasets, MCU boards for Training and Inference on MCUs

### Datasets

We converted the listed datasets into MCU executable *.h* files and placed them inside the Opt-OVO folder, which are used for training multi-class classifiers on MCUs using *Opt-OVO*. The users have to uncomment their dataset of choice (header file at the beginning of the main algorithm program) to use it when training and inference on MCUs.

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

For evaluation, we selected two multi-class datasets using which the Opt-OVO algorithm trains multi-class classifiers on B1-B4. For the first evaluation round, we use the same 64 features Handwritten Digits dataset. Here, we built 3 train sets of various class counts and sizes. For the first train set, we extract data fields corresponding to the handwritten digits 0 to 2 to build a 3 class train set of size 432. The second train set is of class count 5 (digits 0 to 4) and size 720. The last train set of size 1476 contains 10 classes (digits 0 to 9). In all the 3 train sets, each class is of the size 144. 

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

### Onboard Accuracy

The accuracy of the Opt-OVO trained models on MCUs with the train samples used are provided in [Train_time_and_accuracy_results.xlsx](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Train_time_and_accuracy_results.xlsx). We are not presenting the explicit performance comparing of the classifiers trained using Opt-OVO, with the classifiers trained on high resource setups using *Python scikit-learn* since we achieve similar accuracies when experimenting using the same setup and datasets. 