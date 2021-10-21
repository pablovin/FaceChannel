Self-Affective Memory
=======================


.. image:: ../../Images/demo_SelfAffective.png
	:alt: FaceChannel Demo
	:align: center


The Self-Affective Memory is a online learning model that uses the FaceChannelV1 predictions combined with a Growing-When-Required (GWR) network to produce a temporal classification of frames.
It expects that each frame sent to it happens after the previously sent frame. It is able to predict arousal and valence, by reading the average of the current nodes of the GWR.


.. toctree::
    SelfAffectiveMemory.Modules
