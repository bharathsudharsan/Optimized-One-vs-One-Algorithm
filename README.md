# Training up to 50 Class ML Models on Arduino MCUs and Real-time Inference 

In this repo, we present code of Opt-OVO, which is an optimized (resource-friendly) version of the popular One-vs-One algorithm which enables high-performance multi-class ML classifier training and inference directly on microcontroller units (MCUs). We evaluate Opt-OVO by performing live ML model training on 4 popular MCU boards using datasets of varying class counts, sizes and feature dimensions.  

**Exciting finding** On the  3 $ ESP32, Opt-OVO trained a multi-class ML classifier using a dataset of class count 50 and performed unit inference in super real-time of 6.2 ms.