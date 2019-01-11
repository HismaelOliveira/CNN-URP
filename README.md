# Convolutional Neural Network to Usage Related Post

The greatest form of interaction in social systems is through postings, whether public or private, which can deal with various subjects. Several of these posts may be related to the user’s thinking about using the system at that time, for example: <i>"It’s so boring to
have only 140 characters to post on twitter :(".</i> We call these <b>Usage Related posts</b> (URP).
<br/>
This repository contains an automatic way of identificated URP in twitter posts in portuguese using a <b>Convolutional Neural Network</b>. <br/>

## Process
In the database, we remove all the stop words, pontuations and <i>emogis</i>. <br/>
The model applied the convolution filters in the sentences and the dense layer is respondible to identify the class give the resuls of the convolution layer. The process is shown the figure below.
<img src="https://cdn-images-1.medium.com/max/1000/0*0efgxnFIaLTZ2qkY">

This code has <b> 91.5% </b>accuracy and <b> 12.3%</b> of loss. 
<br/><br/>
 Dependencies:
 <ul>
  <li> Python 3.x or python 2.7</li>
  <li> Keras 2.x </li>
  <li> Tensorflow or Theano</li>
  <li> NLTK </li>
  <li> Sklearn </li>
  </ul>
