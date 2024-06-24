**FaceChannel Library - Facial Expression Recognition Models!**

![Screenshot](Images/demo.png)

This repository holds the FaceChannel Library. The FaceChannel contains different facial expression recognition models, and makes it easier to deploy and use them.

**Instalation and Documentation**

You can install the library using pip:

    pip install facechannel

and you can check the [full documentation](https://facechannel.readthedocs.io/en/latest/)  for more information.

Also check the examples folder for a full set of ready-to-use demos!


**Avaliable Models**

FaceChannel is a python library that holds several facial expression recognition models. The main idea behind the FaceChannel is to facilitate the use of this technology
by reducing the deployment effort. This is the current list of available models:



Model | Input Type | Output Type |
------------- | ------------- | -------------
FaceChannelV1  Cat  | Single Image (64x64) | ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"] |
FaceChannelV1  Dim  | Single Image (64x64) | ["Arousal", "Valence"] |
Self Affective Memory  | Single Image (64x64) | ["Arousal", "Valence"] |


* FaceChannelV1 -  [Barros, P., Churamani, N., & Sciutti, A. (2020). The facechannel: A fast and furious deep neural network for facial expression recognition. SN Computer Science, 1(6), 1-10](https://link.springer.com/article/10.1007/s42979-020-00325-6)
* Self Affective Memory - [Barros, P., & Wermter, S. (2017, May). A self-organizing model for affective memory. In 2017 International Joint Conference on Neural Networks (IJCNN) (pp. 31-38). IEEE.](https://www2.informatik.uni-hamburg.de/wtm/publications/2017/BW17/Barros-Affective_Memory_2017-Webpage.pdf)



**License**

All the examples in this repository are distributed under a Non-Comercial license. If you use this environment, you have to agree with the following itens:

- To cite our associated references in any of your publication that make any use of these examples.

- To use the environment for research purpose only.

- To not provide the environment to any second parties.



**Contact**

Pablo Barros - pablo.alvesdebarros@iit.it




