# LSTM

Playing around with various LSTM architectures and figuring out TensorFlow.
Using the [IMDb review data](http://ai.stanford.edu/~amaas/data/sentiment/ "aclIMDb") to do that.

#### Achieved ~85% accuracy. 
# Visualization
The network yields different types of activations on different neurons. The visualizations of the most interesting neurons are shown below. Each image represents the activations of one single neuron in the network. Each row represents one input example (one review). Each column represents the activations of that neuron at time *t*, for *all* the shown input examples.

Neuron shown below fires positively when detects words such as *worst*, *bad*, *awful* and positively when it detects words such as *classic*, *good*, *finest* etc.
It also puts them on a linear scale where some words are more positive/negative than others.
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/sentiment_detector.png "")

This neuron learned to count zeroes:
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/zero_counter.png "")

For comparison, here is a neuron that is not very interesting, which models some uninterpretable features.
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/not_interesting.png "")

The whole data structure (activations of *all* the neurons for *all* the examples for *all* the time steps) could be seen as a 3d tensor from which we're extracting various lower dimensional slices (in this case, abstractions over two variables, i.e., images). 

Inspired by Karpathy's [amazing article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/ "").
