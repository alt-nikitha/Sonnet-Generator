# Sonnet-Generator

The sonnet is a type of poem that has been a part of the literary repertoire since the thirteenth century. 
Sonnets can communicate a sundry of details contained within a single thought, mood, or feeling, typically culminating in the last lines. 

### IMP NOTE:- For the 1st Phase, i.e., text->sonnet -> Use the Heroku link; For Image ->sonnet use any of the ngrok links


## Model 
Image Captioning:
The captioning model is a combination of two separate architecture that is CNN (Convolutional Neural Networks)& RNN (Recurrent Neural Networks) and in this case LSTM (Long Short Term Memory), which is a special kind of RNN that includes a memory cell, in order to maintain the information for a longer period of time. 
CNN is used to generate feature vectors from the spatial data in the images and the vectors are fed through the fully connected linear layer into the RNN architecture in order to generate the sequential data or sequence of words that in the end generate description of an image by applying various image processing techniques to find the patterns in an image.

Sonnet-Generator:
The model is character-based and does not learn the meaning or context of words. A sonnet like any poem is abstract and does not have clearly understandable meaning. For this reason, it is sufficient for the model to learn the sequences of characters alone. With a sequence of 100 characters, it is able to predict a much longer sequence of text with a proper structure. The model uses a special type of Recurrent Neural Network (RNN) called a Long Short Term Memory (LSTM) Netowrk to remember relevant patterns in previous sequences, forget unnecessary details, and predict the next sequence.

## Environment/Tools Used

-Bootstrap 4
-Flask
-Google Colab
-GitHub
-Heroku
-Tensor Flow
 ## Additionally, all the installation packages are present in the requirements.txt file

## Integration of Image Captioning with Sonnet Generator
 
The image caption generator is trained separately, then flask is used to wrap it with the sonnet generator. The caption generated by the image captioning is fed as input to existing sonnet generator model.


### Training

To train the model from scratch, a system with a GPU is required. It is recommended that the files be run on a service that provides GPU support such as Google Colab. To train the model on colab, use the following steps:

(Note: commands mentioned work mostly for OS X or linux OS)

` !git clone https://github.com/alt-nikitha/Sonnet-Generator.git ` 

` cd Sonnet-Generator ` 

` !pip install -r requirements.txt ` 

` !python pre.py dataset.txt dataset2.txt ` 

` !python train.py ` 

Even with a gpu, training can take quite some time, so wait until that's done.
The trained models will get stored in the training_checkpoints folder. At this point, you can zip this folder and download it for future use as follows:


` !zip -r  training_checkpoints.zip  /training_checkpoints/ `

You can also save it to google drive. Mount your google drive:
    
``` 
from google.colab import drive
drive.mount('content/gdrive') 
```

Next copy the folder to your drive:

` cd .. `

` !cp -r /content/training_checkpoints /content/gdrive/My\ Drive `

` cd Sonnet-Generator ` 

### Testing
We have created a Flask application that provides a nice UI to generate sonnets when a user enters a starting prompt. The application uses the trained model in the backend.
We can run this on a local system, so you can head over to the command prompt after you've cloned and changed directory locally. You can also deploy it with a public URL which we have done in two ways- 1.using google colab,and 2. using heroku, which we will get to that in a bit. <br>
On your local system, you can replace the contents of the training_checkpoints folder with your own trained model from the previous training steps or work with the weights we've already saved.

First, create a virtual environment:

`python -m venv name_of_virtualenv`

Replace the name_of_virtualenv with a name of your choice.
Next, activate the virtual environment.

` source name_of_virtualenv/bin/activate `

Finally run this command to run the server for the flask application on your system

` !python app.py `

Once the server is running, enter the local host address that appears on the console into your browser window. <br>
The address will look like this: http://127.0.0.1:5000/
The flask app is now up and running locally, where you can enter a starting phrase to generate sonnets.
    
## Deploying 

### Google Colab

A public URL can be generated by running the same thing on google colab with a few changes to the ` app.py ` file

First, install ngrok library by running

` !pip install flask-ngrok`

ngrok is a lightweight tool used to generate public URLs for testing your model.

Second, make sure in your app.py file, the function app.run(), does not have debug set to True.

Third, in your app.py file, import this:-

`from flask_ngrok import run_with_ngrok`

Fourth, in your app.py file, 
type in this command:- 

`run_with_ngrok(app)` right below this 

`app = Flask(__name__,static_url_path='/static')`(Save all these changes in app.py)

Finally, run app.py

`!python app.py`

Click on the url that appears to see the app that's running.

The deployed App URL for image captioning and sonnet generation:  
http://2309a8e18bd5.ngrok.io

Second Link: http://85604abaf200.ngrok.io/
Can try any!
### Note: Ensure one person tries at a time!

### Heroku
(Only for Phase I-Sonnet Generation through text)
This has to be done on your local terminal. <br>
Ensure that the app.py has ` debug=False `
Next create an account on heroku. Once this is done, on your local terminal, install the gunicorn dependency, which is a Python Web Server Gateway Interface HTTP server. Ensure your virtual environment is activated.

` pip install gunicorn `

Next login to heroku on your terminal. 
` heroku login `

We need to now ensure that heroku installs all the libraries and dependencies to run and host our model. So we run ` pip freeze > requirements.txt ` so that all the dependencies get stored.

We need to now create a Procfile, which tells heroku what processes or commands it should run. Since we need to run a web process for app.py, enter ` web: gunicorn app:app ` in the Procfile you've created in the directory of your project.

Next, we create an app with a custom name

` heroku create app_name ` 

We chose sonnet-generator. After the app is created, push all the files in your repo to a remote called heroku.

` git add . `

` git commit -m "commit message" `

` git push heroku master `

The code now gets deployed and a url is generated for your app. This is our URL - https://sonnet-generator.herokuapp.com/

Finally, make sure atleast one instance of the app is running : <br>

` heroku ps:scale web=1 `

Now you can either directly use the URL, or open the app through your command-line by typing ` heroku open `


## References :

The model architecture was based on the framework provided in these links. Specific parameters were altered to adjust for sonnets and also the layers in the models were changed for this specific purpose.  

https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/

https://www.tensorflow.org/tutorials/text/text_generation

http://karpathy.github.io/2015/05/21/rnn-effectiveness/
