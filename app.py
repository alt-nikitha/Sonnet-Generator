from flask import Flask, render_template, request 
import tensorflow as tf 
import numpy as np 
import os  
import train as t
import generate as gen


model = tf.keras.models.load_model('model') 
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

app = Flask(__name__) 


@app.route('/index',methods=['post', 'get'])
def index():
	if request.method == 'POST':

        prefix = request.form.get('prefix')  # access the data inside

        return render_template('index.html',text=gen.generate_text(model,prefix))
    else:
        return render_template('index.html',text="Enter something pls")


if __name__ == "__main__":
    
    app.run(debug=True)


