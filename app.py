from flask import Flask, render_template, request 
import tensorflow as tf 
import numpy as np 
import os  
import generate as gen

#to define model architecture so weights can be loaded accordingly
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model
#path to preprocessed dataset
path_to_file = 'preprocess.txt'

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#unique characters in the preprocessed text
vocab = sorted(set(text))
#dictionary to map each character to a numeric index
char2idx = {u:i for i, u in enumerate(vocab)}
#to map index to character
idx2char = np.array(vocab)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
#file for pretrained model
checkpoint_file = './training_checkpoints/ckpt_9'

#create the skeleton for the model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

#store weights
model.load_weights(checkpoint_file)

#create an instance of the flask app
app = Flask(__name__,static_url_path='/static') 

#what happens in the index.html page
@app.route('/index',methods=['post', 'get'])
def index():
  #if input in the form by post method
  if request.method == 'POST':
    #get the prefix
    prefix = request.form.get('prefix')
    #render the index.html page with the sonnet after it is generated
    return render_template('index.html',text=gen.generate_text(model,prefix.lower(),0.5,char2idx,idx2char))
  else:
    #if no input is given 
    return render_template('index.html',text="")

@app.route('/')
def rootfile():
  #root url, index.html with no text in the textarea
  return render_template('index.html',text="")
    
if __name__ == "__main__":
    #run the app, set debug=True during testing
    app.run(debug=False)


