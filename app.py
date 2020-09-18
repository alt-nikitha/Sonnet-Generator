from flask import Flask, render_template, request 
import tensorflow as tf 
import numpy as np 
import os  
import generate as gen

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
path_to_file = 'preprocess.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

checkpoint_file = './training_checkpoints/ckpt_10'
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(checkpoint_file)

app = Flask(__name__,static_url_path='/static') 


@app.route('/index',methods=['post', 'get'])
def index():
    if request.method == 'POST':
        prefix = request.form.get('prefix')
        # temperature = float(request.form.get('temperature'))
        temperature=0.5
        return render_template('index.html',text=gen.generate_text(model,prefix,temperature,char2idx,idx2char))
    else:
        return render_template('index.html',text="")


if __name__ == "__main__":
    
    app.run(debug=True)


