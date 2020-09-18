
import tensorflow as tf
import numpy as np
import os
import time


def generate_text(model,start_string,temperature,char2idx,idx2char):
  # To generate text using the loaded model

  # Number of characters to generate
  num_generate = 500

  # Convert the start string to a vector of numbers
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # To store the results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text. 
  
  temperature = temperature

  # Here batch size == 1
  model.reset_states()
  i=0
  while(i<num_generate or idx2char[predicted_id]!=' '):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # The predicted character is passed as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])
    i+=1
  return (start_string + ''.join(text_generated))

  
  

  # checkpoint_file = './training_checkpoints/ckpt_10'

  # model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

  # model.load_weights(checkpoint_file)

  # model.build(tf.TensorShape([1, None]))

  # model.summary()

  # return(generate_text(model, start_string=prefix))
