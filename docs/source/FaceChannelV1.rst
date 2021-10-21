FaceChannelV1
=======================


.. image:: ../../Images/FaceChannel_v2.png
	:alt: FaceChannel Demo
	:align: center


FaceChannelV1 is a smaller version of the FaceChannel model, with 800 thousand parameters. It was trained on the FER+ dataset, and for the dimensional version fine-tuned on the AffectNet dataset. It is available in two types: Cat, for categorical output with 8 different emotions, and Dim, for a dimensional output representing arousal and valence. FaceChannelV1 works on a frame-level, so for every input frame, it produces one output.


.. toctree::
    FaceChannelV1.Modules
    FaceChannelV1.ImageProcessing
