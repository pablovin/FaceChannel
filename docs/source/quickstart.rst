Quickstart Guide
================

Here you will find instructions regarding how to install the library and run your first demo!

Instalation
^^^^^^^^^^^

To install the FaceChannel library, you will need python >= 3.6. The environment has a list of `requirements <https://pypi.org/project/facechannel/>`_ that will be installed automatically if you run:

.. code-block:: bash

    pip install facechannel


Understanding FaceChannel
^^^^^^^^^^^^^^^^^^^^^^^^

FaceChannel is a python library that holds several facial expression recognition models. The main idea behind the FaceChannel is to facilitate the use of this technology
by reducing the deployment effort. This is the current list of available models:

.. list-table:: Title
   :widths: 25 25 50
   :header-rows: 1

   * - Model
     - Input Type
     - Output Type
   * - FaceChannelV1 - Cat
     - (64x64x1)
     - ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]
   * - FaceChannelV1 - Dim
     - (64x64x1)
     - ["Arousal", "Valence"]
   * - Self Affective Memory
     - (64x64x1)
     - ["Arousal", "Valence"]


* FaceChannelV1 -  `Barros, P., Churamani, N., & Sciutti, A. (2020). The facechannel: A fast and furious deep neural network for facial expression recognition. SN Computer Science, 1(6), 1-10. <https://link.springer.com/article/10.1007/s42979-020-00325-6>`_
* Self Affective Memory - `Barros, P., & Wermter, S. (2017, May). A self-organizing model for affective memory. In 2017 International Joint Conference on Neural Networks (IJCNN) (pp. 31-38). IEEE. <https://www2.informatik.uni-hamburg.de/wtm/publications/2017/BW17/Barros-Affective_Memory_2017-Webpage.pdf>`_


Recognizing Facial Expression
^^^^^^^^^^^^^^^^^^^^^

To start the facial expression recognition is simple and painless:

.. code-block:: python

    """Facial Expression Recognition"""
    from FaceChannel.FaceChannelV1.FaceChannelV1 import FaceChannelV1

    faceChannelCat = FaceChannelV1("Cat", loadModel=True)

    categoricalRecognition = faceChannelCat.predict("image.png")

    print categoricalRecognition

For more examples on how to use, and to see our pre-made demos, check the examples folder.
