import tensorflow as tf

import numpy as np
import os
import time

# 
def split_input_target(chunk):
  '''
  function to create a mapping of input text, and target text to be generated
  input text is the text chunk till penultimate character. The target text 
  is what has to be predicted sequentially from the first character in the 
  text. So it is the set of characters in the text chunk, excluding 
  first character and including the last character to be predicted

  '''
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  '''
  Define an embedding layer which is the input to the model. The number that
  is mapped to each character is further mapped to a vector representation with
  embedding_dim dimensions. This layer is trainable

  Next, define a GRU layer in which each unit is responsible for taking 
  characters sequentially, predicting the next character based on stored 
  relevant information from previous units, and passes the predicted character to the subsequent unit.
  The returned sequence is passed through a dense layer to returns an output of
  vocab size, with probability for each char in the vocabulary
  '''
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def train_step(inp, target,optimizer):
  '''
  To calculate predictions and loss.
  Then the gradients of the loss with respect to model variables are calculated.
  Next use optimiser to apply gradients.
  '''
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss




#raw dataset
path_to_file = "preprocess.txt" # path to preprocessed file

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))



# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# Creating a dictionary that maps unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

#encode text as integers
text_as_int = np.array([char2idx[c] for c in text])



# The maximum length sentence we want for a single input in characters
seq_length = 50
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

'''create batches of seq_length characters that can be divided into overlapping
input and target text'''
sequences = char_dataset.batch(seq_length+1, drop_remainder=True) 

# for item in sequences.take(5):
#   print(repr(''.join(idx2char[item.numpy()])))


#creates input and target texts using function defined

dataset = sequences.map(split_input_target) 

# #look at one example of input and target
# for input_example, target_example in  dataset.take(1):
#   print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#   print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))



# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# print(dataset)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

#instantiate a model with the architecture defined
model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()

# define optimizer and loss function for model
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

#to save checkpoints during training
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

## advanced customised training


model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()

# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  # resetting the hidden state at the start of every epoch
  model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # saving (checkpoint) the model for every epoch
  
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))



