# Subclass Layering or Modeling?
Effectively, the Layer class corresponds to what we refer to in the literature as a "layer" (as in "convolution layer" 
or "recurrent layer") or as a "block" (as in "ResNet block" or "Inception block").
Meanwhile, the Model class corresponds to what is referred to in the literature as a "model"
(as in "deep learning model") or as a "network" (as in "deep neural network").

So if you're wondering, "should I use the Layer class or the Model class?", ask yourself: will I need to call fit() on it?
Will I need to call save() on it? If so, go with Model. If not (either because your class is just a block in a bigger 
system, or because you are writing training & saving code yourself), use Layer.

For more info for subclass layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer 


# Definition of ConvBlk, TDBlk, TUBlk
Dense CNNs first got noticed from [Densely Connected Convolutional Neural Network](https://arxiv.org/pdf/1608.06993.pdf)

This paper defines these blocks as follow:

BN-ReLU-Conv1-BN-ReLU-Conv3

However, we we aimed to implement the 
[The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
paper which defines the blocks as follow:

ConvBlk: BN - Relu - Conv 3x3 - Dropout 0.2

TD (Transition Down): BN - Relu - Conv 1x1 - Dropout 0.2 - Maxpooling (2 by 2) 

TU (Transition UP): 3x3 Transposed Conv stride=2

# Architecture

In Table 3, we report our results for three networks with
respectively (1) 56 layers (FC-DenseNet56), with 4 layers
per dense block and a growth rate of 12; (2) 67 layers (FCDenseNet67) with 5 layers per dense block and a growth
rate of 16; and (3) 103 layers (FC-DenseNet103) with a
growth rate k = 16 (see Table 2 for details).