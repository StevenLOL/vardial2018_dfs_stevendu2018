Abstract 

This paper introduces the submitted system for team STEVENDU2018 during VarDial 2018 Discriminating between Dutch and Flemish in Subtitles(DFS). Post evaluation analyses are also presented. The results obtained indicate that it is a challenging task to discriminate between Dutch and Flemish.


Introduction 

The DFS task is a supervised learning task to classify text into Dutch or Flemish. Dutch is the language spoken in the Netherlands and Flemish is a variant of Dutch language and also known as Belgian Dutch. There are 300000 labeled training data, 500 labeled development data, 20000 on-hold test data. DUT in training labels denotes Dutch, and BEL is the label for Flemish. F1 score is the evaluation metrics. 

This paper is structured as follows: first, a brief training data analysis will be given. Then systems trained during the evaluation will be introduced. Finally more systems will be explored for post evaluation analysis.

Data analysis 

The training data set consists of 300000 labeled sentences. After being lower cased and tokenized, the average sentence's length in characters and number of words for both DUT and BEL is nearly the same. As showed in Table 1, it is a well balanced data set. It is worth to note that the two languages share 57.2% of vocabulary.


One interesting finding is that the use of punctuation is a little bit different. BEL has more commas, periods and question marks but less exclamation marks than DUT as showed in Table 2.
  
Statistics for the punctuation in the training data set.

Systems trained during evaluation

There are two systems trained during evaluation: a bag-of-ngram model and dual convolutional neural network model.


Bag-of-ngram

Conventional methods for text classification apply common features such as bag-of-words, n-grams, and their TF-IDF features (Zhang et al., 2008) as input of machine learning algorithms such as support vector machine (SVM) (Joachims, 1998), logistic regression (Genkin et al., 2007), naive Bayes (NB) (Mccallum, 1998). 

In this work, the bag-of-ngram system and Linear SVM are used as the baseline system. First the text is lower-cased and converted to n-gram tokens (n is from 1 to 3), then filtered by TF-IDF with minimal document frequency of 5. Extracted features are utilized to train a linear SVM classifier. A 20 folds cross validation is performed on the training set, the average F1 score is 0.63 and 0.69 is obtained on the development set.

Dual-CNN

This approach builds a simple CNN model (with pre-trained embedding) for each language. The input text will pass through these CNNs separately. Outputs of two CNN networks are then concatenated together. This is followed by a fully connected layer for the classification task. Detail of this network can be found at github . During evaluation the proposed Dual-CNN network obtains 0.62 through cross validation and score 0.61 on the development set. 

The final submitted system is only a bag-of-ngram model which has better performance than the DualCNN.

Evaluation results

The score on the released test set range from 0.55 to 0.66 in Table 3, our bag-of-ngram, the most simple approach yields 0.623. On the other hand proposed Dual-CNN yields 0.621. The test score correlated well with the local cross validation score, development set is not the right choice for model selection. The best score is just 0.66, which implies that the DFS task is challenging.


Post evaluation systems

Since bag-of-ngram system only scores 0.623 on test set, to achieve better result a series of studies had been carried out after the evaluation. These can be broadly divided into three groups: one group focus on finding the vector representation for the given text data, another group focus on deep learning approaches, third group utilize existing text classification framework.


Vector representation based approach

Vector representation approach intends to convert text data in variable-length pieces of text into a fixed-length low dimension vector. There are many works have been done in this direction (Kim, 2014; Wieting et al., 2015; Kusner et al., 2015; Kenter et al., 2016; Ye et al., 2017), but only two basic approaches are investigated here: by taking mean value of word vectors and through doc2vec from the work in distributed representation of sentences and documents (Le and Mikolov, 2014).

Mean word vector system 

A popular idea in modern machine learning is to represent words by vectors. These vectors capture hidden information about a language, like word analogies or semantic. Commonly used word vector is word2vec (Mikolov et al., 2013), Glove (Pennington et al., 2014) and fastText (Bojanowski et al., 2017). Compare to word2vec, FastText is capable to capture sub-word information, thus in this study, we use FastText to train word vectors. Skip-gram, window size of 5 and minimal word count of 5, 5 negative samples, sub-word range is between 3 and 6 characters are the default training parameters. After training, for each sentence, the mean value of its word vectors is used as feature, Linear Discriminant Analysis classifier is selected as the classifier.

Table 1 shows F1 score for the mean word vector system. With increase in the length of word vectors, the system performance better. The 400 dimensional word vector is suitable for this task.



Doc2vec

In this study, we use the doc2vec (Le and Mikolov, 2014) from gensim3. The doc2vec model is trained on training data set with minimal word occurrence of 5 and window size of 8. Table 5 shows the best score is 0.5308, which is slightly better than random guess.

Two sets of sentence vector had been evaluated in this study. The average word vector approach is better than doc2vec. In the following experiment, 400 is used as the default size of word embedding.

Our proposed Dual-CNN didn’t beat the conventional bag-of-ngram model. This motivated us to examine the performance of deep learning approaches. Five types of deep learning based approaches are investigated, started from the most basic architecture, they are:

The MLP system is built by an embedding layer, one flatten layer and fully connected layer. Please refer to system diagrams in github repository.



















s

The Average system is similar to MLP system but the flatten layer is replaced by an average pooling layer. It is also known as neural bag-of-word model and being surprisingly effective for many tasks (Iyyer et al., 2015).

The GRU system is similar to AVERAGE system but the average pooling layer is replaced by a bidirectional GRU layer.

The CNN-LSTM system is build by an embedding layer followed by two convolution-max pooling layers and one bidirectional GRU layer. 

The four deep approaches are indeed most fundamental networks in NLP research. Incorporating language model fine-tunning (Howard and Ruder, 2018) and attention mechanism (Vaswani et al., 2017) is the recent trends, which we leave it for further exploration.

Table 6 presents the result of four popular deep learning based approaches. D20 Random denotes randomized word embedding of 20 dimensions is used in the network. D400 pre-trained denotes embedding layer is pre-trained with word vector size of 400 dimensions. This result confirms the observation in 4.1.1, that 400 dimension word vectors is a good choice for this task. Three out of four systems are higher than 0.64 which are significant better than submitted baseline system

Capsules with transformation matrices allowed networks to automatically learn part-whole relationships. Consequently, (Sabour et al., 2017) proposed capsule networks that replaced the scalar-output feature detectors of CNNs with vector-output capsules and max-pooling with routing-by-agreement. The capsule network has shown its potential by achieving a state-of-the-art result on highly overlapping digit parts in MutiMNIST data set. The PrimaryCapsule used in that paper is a convolutional capsule layer with 32 channels of convolutional 8D capsules. We increase the number of channels from 32 to 320 in this study, the assumption is that there are more part-whole relations in language than those in MNIST digit images.

Table 7 introduces F1 score of CapsuleNet on the test data set. The results indicate that with increase of number of channels and thus the number of capsules the system performance better. When changing the binary classification problem to two class classification problem, the capsule net yield comparable result to the bag-of-ngram baseline. The work by (Zhao et al., 2018) also shows significant improvement when transferring single-label to multi-label text classifications.

Text Classification Framework

FastText (Joulin et al., 2016) is a library for efficient learning of word representations and sentence classification5 . It use vectors to represent word n-grams to take into account local word order, which is important for many text classification problems. Following Table 8 shows fastText classification results. The 0.6476 is the highest score achieved
Table 8: FastText Classification results. The 0.6476 is the highest score achieved.

Conclusion

In this paper, a wide range of systems have been evaluated for the ValDial 2018 DFS task. A bag of-ngram system score 0.6230 and serves as the baseline. Complex systems such as Dual-CNN and CapusleNet have competitive score to baseline system. Four simple deep learning based methods outperform baseline, three of them are higher than 0.64. FastText is identified as the best single system, yielded a F1 score of 0.6476.
