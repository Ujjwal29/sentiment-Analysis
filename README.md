## sentiment-Analysis

I took up this project to understand the inner working of text data using Natural Language Processing.
This repo contains code for the famous problem in NLP that is sentiment analysis.
The problem statement deals with contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations. Brands can use this data to measure the success of their products in an objective manner. The problem is taken from a bunch of competitions running on Analytics Vidhya. You can find more about the problem statement [here](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/). Till now, I have used

  - Pandas for reading the file and numpy for matrix calculations
  
  - [NLTK](https://www.nltk.org/) library for data pre-processing
  
  - NLTK library for stopwords removal, stemming the data, tokenization
  
  - wordcloud library in python for displaying the wordcloud
  
  - [Scikit learn](https://scikit-learn.org/) library for Bag of words and Tf-Idf model
  
  - Logistic Regression classifier from scikit-learn to classify the given data using both Bag or words and Tf-Idf model
  
Using the above steps, I am getting an f1_score of around 0.76 on both train and test data. The project is still uner progress and I'll keep updating the model and aim for a better f1_score

Also, I'll keep updating the README file as an when the project progress.
Futher steps to be done:
  - Pre-process the data even more to build a better model by using steps like lemmetization
  
  - Further EDA on data to get detailed insight about each class
  
  - Use [gensim](https://pypi.org/project/gensim/) library to build more robust word embeddings like [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) and [GloVe](https://radimrehurek.com/gensim/scripts/glove2word2vec.html)
  
  - Use robust Machine Learning models like Random Forest, SVM, XgBoot to classify the text
  
  - Use deep learning models like ANN to aim for 90-95% accuracy

## Edit

The new solution in *solutin2.py* is an extended work by using the function from [SivaAndMe](https://github.com/SivaAndMe/Sentiment-Analysis-on-Swachh-Bharat-using-Twitter/blob/master/swn_sentiment_labeling.py). Note that the results after running the new solution are not very great and depreciated from the previous results.
