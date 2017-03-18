# LSTM

Playing around with various LSTM architectures and figuring out TensorFlow.
Using the [IMDb review data](http://ai.stanford.edu/~amaas/data/sentiment/ "aclIMDb") to do that.

#### Achieved ~85% accuracy. 
# Visualizations
The network yields different types of activations on different neurons. The visualizations of the most interesting neurons are shown below.

Neuron that fires positively when detects words such as *worst*, *bad*, *awful* and positively when it detects words such as *classic*, *good*, *finest* etc.
It also puts them on a linear scale where some words are more positive/negative than others.
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/sentiment_detector.png "")
---
Neuron that learned to count zeroes:
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/zero_counter.png "")
---
For comparison, here is a not very interesting neuron, which models some uninterpretable features.
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/not_interesting.png "")





Inspired by Karpathy's [amazing article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/ "").
