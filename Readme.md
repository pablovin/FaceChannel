**Frame-Based Emotion Categorization**



This repository holds the code for the FaceChannel neural network: a light-weight neural network for automatic facial expression recognition that has much fewer parameters than common deep neural networks. 


**Train Model**


![Screenshot](Images/FaceChannel_v2.png)

The trainModel.py holds the model's architecture, for both categorical and dimensional outputs.
Also, it holds a simple training scheme.



**Demo**

![Screenshot](Images/demo.png)

This demos is configured to run using two different models: one for categorical emotions and other for arousal/valence intervals.


Both models are implemented with KERAS. more information can be found here: <br>



****Requirements****

Install all the libraries on the requirements.txt file.

****Instructions****


To run the demo with your own model (has to be saved as a KERAS model), add an entry on the modelDictionary.py containing the model's directory, class dictionary and type. Also, change the run.py to matche your inputsize (faceSize).


The run.py file contains all the necessary configurations. This demos runs on Python 3.x.


To run the demo just use
```sh
$ python run.py

```

**Related Publications**


```sh
P. Barros, N. Churamani and A. Sciutti,  "The FaceChannel: A Light-Weight Deep Neural Network for Facial Expression Recognition.," in 2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020) (FG), Buenos Aires, undefined, AR, 2020 pp. 449-453.
doi: 10.1109/FG47880.2020.00070
keywords: {emotion recognition;deep learning}
url: https://doi.ieeecomputersociety.org/10.1109/FG47880.2020.00070

```


**License**

All the examples in this repository are distributed under the Creative Commons CC BY-NC-SA 3.0 DE license. If you use this corpus, you have to agree with the following itens:

- To cite our reference in any of your publication that make any use of these examples. The references are provided at the end of this page.
- To use this model for research purpose only.


**Contact**

Pablo Barros - pablo.alvesdebarros@iit.it




