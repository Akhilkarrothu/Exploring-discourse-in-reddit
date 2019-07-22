# Authors : Akhil Karrothu, Rishab Kata
# Foundations of intelligent Systems
# Project 2 
import requests
import time
import pandas as pd
import nltk
import ssl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")

def subredditone():
	# declaring the bot that downloads our posts
	headers = {'User-agent': 'E Bot 1.0'}  
	after = None  # after parameter to get next 25 posts every time
	unpopular_opinion = []
	url = "https://www.reddit.com/r/unpopularopinion/.json" # assigning the subreddit url.

	# get 500 posts by getting 25 posts everytime in a loop
	for i in range(20):
	    print(i)
	    if after == None:  # after is none for the first page
	        params = {}
	    else:
	        params = {'after': after}  # if not first page, change after
	    res = requests.get(url, params=params, headers=headers)  # sending the request to reddit to get posts

	    if res.status_code == 200:  # 200 code means request got processed successfully
	        unpopular_opinion_json = res.json()  # get the posts in json format
	        unpopular_opinion.extend(unpopular_opinion_json['data']['children'])  # add posts to the list
	        after = unpopular_opinion_json['data']['after']  # change after to the next set of posts.
	    else:
	        print(res.status_code)
	        break
	    time.sleep(2)  # sleep 2 seconds before pinging the server again.

	unpopular_opinion_list = [unpopular_opinion[i]['data']['title'] + ' ' +
	                          unpopular_opinion[i]['data']['selftext']
	                          for i in range(2, len(unpopular_opinion))]
	uo_df = pd.DataFrame(unpopular_opinion_list, columns=['post'])  # creating a dataframe from the list.
	uo_df['subreddit'] = 'unpopular_opinion'
	uo_df.head()
	subreddittwo(uo_df)

def subreddittwo(uo_df):
	headers = {'User-agent': 'E Bot 1.0'}  # changing the user-agent
	after = None  # after parameter to get next 25 posts every time
	world_news = []
	url = "https://www.reddit.com/r/worldnews/.json" # assigning the subreddit url.
	# get 500 posts by getting 25 posts everytime in a loop
	for i in range(20):
	    print(i)
	    if after == None:  # after is none for the first page
	        params = {}
	    else:
	        params = {'after': after}  # if not first page, change after
	    res = requests.get(url, params=params, headers=headers)  # sending the request to reddit to get posts

	    if res.status_code == 200:  # 200 code means request got processed successfully
	        world_news_json = res.json()  # get the posts in json format
	        world_news.extend(world_news_json['data']['children'])  # add posts to the list
	        after = world_news_json['data']['after']  # change after to the next set of posts.
	    else:
	        print(res.status_code)
	        break
	    time.sleep(2)  # sleep 2 seconds before pinging the server again.

	world_news_list = [world_news[i]['data']['title'] + ' ' +
	                   world_news[i]['data']['selftext']
	                   for i in range(2, len(world_news))]
	wn_df = pd.DataFrame(world_news_list, columns=['post'])  # creating a dataframe from the list.
	wn_df['subreddit'] = 'world_news'
	wn_df.head()
	combinedata(uo_df,wn_df)

def combinedata(uo_df,wn_df):	
	df = uo_df.append(wn_df, ignore_index=True)  # appending two dataframes and create a new dataframe
	datatransformatin(df)

def datatransformatin(df):
	X = df ['post']# Assigning the columns in the dataframe to X and Y
	y = df['subreddit']
	df['target'] = df.subreddit.astype('category').cat.codes# Making a column target to represent the subreddit column in the integer or float 
	num_class = len(np.unique(df.subreddit.values))# checking for the category of the classification
	y = df['target'].values
	worddistribution(df,y,num_class)
	

def worddistribution(df,y,num_class):
	df['num_words'] = df.post.apply(lambda x : len(x.split()))
	bins=[0,50,75, np.inf]
	df['bins']=pd.cut(df.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])
	word_distribution = df.groupby('bins').size().reset_index().rename(columns={0:'counts'})
	sns.barplot(x='bins', y='counts', data=word_distribution).set_title("Word distribution per bin")
	time.sleep(6)
	convertion_to_tokens(df,y,num_class)

def convertion_to_tokens(df,y,num_class):
	MAX_LENGTH = 500# After checking out the word distribution we figured sequence length to be 500
	tokenizer = Tokenizer() # intializing tokenizer
	tokenizer.fit_on_texts(df.post.values) #converting text to tokens
	post_seq = tokenizer.texts_to_sequences(df.post.values) # conerting sequences into number
	post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH) # padding the squences
	datasplitting(post_seq_padded,y,MAX_LENGTH,tokenizer,num_class)

def datasplitting(post_seq_padded,y,MAX_LENGTH,tokenizer,num_class):
	X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.2)# Splitting the data into testing and train.
	modelconstruction(MAX_LENGTH,tokenizer,X_train,X_test,y_train,y_test,num_class)

def modelconstruction(MAX_LENGTH,tokenizer,X_train,X_test,y_train,y_test,num_class):
	vocab_size = len(tokenizer.word_index) + 1# calculating vocab size.
	inputs = Input(shape=(MAX_LENGTH, ))# LSTM model construction and training.
	embedding_layer = Embedding(vocab_size,
	                            128,
	                            input_length=MAX_LENGTH)(inputs)# constructing the first hidden layer.
	x = LSTM(64)(embedding_layer)#joining the embedding layer
	x = Dense(32, activation='relu')(x)# First dense layer
	predictions = Dense(num_class, activation='softmax')(x)# formation of second dense layer
	model = Model(inputs=[inputs], outputs=predictions)# model generation
	model.compile(optimizer='adam',
	              loss='binary_crossentropy',
	              metrics=['acc'])# Compile the model
	model.summary()# generating the summary
	filepath="weights.hdf5"# defining the filepath
	checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')# generating the model check point
	history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
	          shuffle=True, epochs=10, callbacks=[checkpointer])# fitting the model
	model.load_weights('weights.hdf5')# loading the stored weights to predict.
	predicted = model.predict(X_test)# performing prediction on the test data.
	predicted = np.argmax(predicted, axis=1)# converting the obtained output
	accuracy = accuracy_score(y_test, predicted)# Accuracy calculator
	print("accuracy of this model is :")
	print(accuracy)
	graphplotting(history)
    
def graphplotting(history):
	print("Graph for epochs vs validation_accuracy :")
	df1 = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})# Constructing new dataframe for storing accuracy and validation accuracy.
	g = sns.pointplot(x="epochs", y="accuracy", data=df1, fit_reg=False)# generating graph for epochs vs accuracy
	g = sns.pointplot(x="epochs", y="validation_accuracy", data=df1, fit_reg=False, color='green')# generating graph for epochs vs validation accuracy.


if __name__ == "__main__":
    subredditone()

