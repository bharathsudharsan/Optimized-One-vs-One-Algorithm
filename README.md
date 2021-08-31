# Training up to 50 Class ML Models on Arduino MCUs and Real-time Inference 

In this repo, we provide the code of Opt-OVO, which is an optimized (resource-friendly) version of the popular One-vs-One algorithm which enables high-performance multi-class ML classifier training and inference directly on microcontroller units (MCUs). We evaluate Opt-OVO by performing live ML model training on 4 popular MCU boards using datasets of varying class counts, sizes and feature dimensions.  

**Exciting finding** On the  3 $ ESP32, Opt-OVO trained a multi-class ML classifier using a dataset of class count 50 and performed unit inference in super real-time of 6.2 ms.

## Datasets, MCU boards for Training and Inference on MCUs

### Datasets

We converted all the listed datasets into MCU executable *.h* files and placed them inside the algorithm folders. Datasets 1, 2, 3, and 4 are used for training binary classifiers on MCUs using *Opt-SGD*. Datasets 5 and 6 are used for training multi-class classifiers on MCUs using *Opt-OVO*. The users have to uncomment their dataset of choice (header file at the beginning of the main algorithm program) to use it when training and inference on MCUs.

4. [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/) (64 features, 10 classes, 1797 samples): Data for each digits 0 to 9 is a class. The onboard *Opt-OVO* trained multi-class classifier should distinguish digits, based on the input features.
6. [Australian Sign Language signs](https://archive.ics.uci.edu/ml/datasets/Australian+Sign+Language+signs+(High+Quality)) (22 features, 95 classes, 6650 samples): Here, the class count is 50 since we extracted the files that contain data of 50 Auslan signs varying from *alive* to *more*. Then using *Opt-OVO*, we trained a classifier on MCUs, that distinguish Auslan signs based on the input features.

### MCU Boards

Using Arduino IDE, we upload the *Opt-OVO* algorithm along with the selected/uncommented dataset on the following popular boards. After successful upload, we trained various ML classifier models on MCUs, performed the onboard model evaluation and inference performance evaluation of the thus trained MCU models.

1. B1 [Generic ESP32](https://www.espressif.com/en/products/devkits): Xtensa LX6 @240MHz, 4MB Flash, 520KB SRAM.
2. B2 [ATSAMD21G18 Adafruit METRO](https://www.adafruit.com/product/3505): ARM Cortex-M0+ @48 MHz, 256kB Flash, 32KB SRAM. 
3. B3 [STM32f103c8 Blue Pill](https://stm32-base.org/boards/STM32F103C8T6-Blue-Pill.html): ARM Cortex-M0 @72MHz, 128KB Flash, 20KB SRAM.
4. B4 [nRF52840 Adafruit Feather](https://www.adafruit.com/product/4062): ARM Cortex-M4 @64MHz, 1MB Flash, 256KB SRAM.

## Opt-OVO Performance Evaluation

We uploaded the *Opt-OVO* algorithm's C++ implementation on all boards. We then power on each board, connect them to a PC via the serial port to feed the training data, receive training time and classification accuracy from MCUs. The first 70% of data was used for training, the remaining 30% data for evaluation. When we instruct the board to train, *Opt-OVO* iteratively loads the data chunks and trains. Next, we load the test set, infer using the trained models to evaluate the MCU-trained classifiers. 

### Training Results

![alt text](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Train_time_results.png)

### Inference Results

![alt text](https://github.com/bharathsudharsan/Optimized-One-vs-One-Algorithm/blob/main/Infer_time_results.png)