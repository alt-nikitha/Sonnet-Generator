from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf 
import numpy as np 
import os  
import generate as gen
import generate_caption as g1

#to define model architecture so weights can be loaded accordingly
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.2),
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
rnn_units = 700
#file for pretrained model
checkpoint_file = './training_checkpoints/ckpt_36'

#create the skeleton for the model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

#store weights
model.load_weights(checkpoint_file)
UPLOAD_FOLDER='./stored_images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
#create an instance of the flask app
app = Flask(__name__,static_url_path='/static') 
app.config['SECRET_KEY'] = '\x13\xa6U\x97\xe0t\x8e\xf0\x0f\xab[\xb4'    

# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS']=['png', 'jpg', 'jpeg']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
  return '.' in filename and \
          filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#what happens in the index.html page
@app.route('/index',methods=['post', 'get'])
def index():
  #if input in the form by post method
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file1 = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file1.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file1 and allowed_file(file1.filename):
        filename = secure_filename(file1.filename)
        path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file1.save(path)
        caption=g1.caps(path)
    # return render_template('index.html',text=path)
    return render_template('index.html',text=gen.generate_text(model,caption.lower(),0.2,char2idx,idx2char))
  else:
    #if no input is given 
    return render_template('index.html',text="")

@app.route('/')
def rootfile():
  #root url, index.html with no text in the textarea
  return render_template('index.html',text="")
    
if __name__ == "__main__":
    #run the app, set debug=True during testing
    app.run(debug=True)


