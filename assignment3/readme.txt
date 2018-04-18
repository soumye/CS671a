Name - Soumye Singhal
Roll No - 150728
NLP Assignment 3

Accuracy : The Accuracy that I am getting for the model is 79%
######################### Observations ##########################################################################################
359781/359781 [==============================] - 58s 162us/step - loss: 0.4786 - acc: 0.7493 - val_loss: 0.4331 - val_acc: 0.7680
Epoch 2/3
359781/359781 [==============================] - 56s 156us/step - loss: 0.4414 - acc: 0.7710 - val_loss: 0.4181 - val_acc: 0.7797
Epoch 3/3
359781/359781 [==============================] - 57s 158us/step - loss: 0.4297 - acc: 0.7781 - val_loss: 0.4094 - val_acc: 0.7810
45866/45866 [==============================] - 2s 42us/step
Accuracy of the Model : 0.79 
######################### Observations ##########################################################################################

Description of the Code 
1. We implemented dataloader to load the dataset of the EWT treebank
2. We then convert the parsed sentnces to their dependency parsed configuration.
3. We then convert the configuration to a feature vector 
4. Then we construct the dataset.
5. Then we contruct a 

Feature vector
For the current configuration we construct the feature vector as follows. We choose the following 7 - word attributes
1. stack top
2,3 first 2 elements of the buffer
4,5 left-right dependency of stack top
6,7 left-right dependecy of first element of buffer
We then concatenated the POS tag with the word2vec of them to form the feature vector we use to predict the next transition.
