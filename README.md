# Sonnet-Generator

The sonnet is a type of poem that has been a part of the literary repertoire since the thirteenth century. 
Sonnets can communicate a sundry of details contained within a single thought, mood, or feeling, typically culminating in the last lines. 

## Model 
The model is character-based and does not learn the meaning or context of words. A sonnet like any poem is abstract and does not have clearly understandable meaning. For this reason, it is sufficient for the model to learn the sequences of characters alone. With a sequence of 100 characters, it is able to predict a much longer sequence of text with a proper structure. The model uses hidden Long Short Term Memory (LSTM) layers to remember relevant patterns in previous sequences and predict the next sequence. 

### Training

To train the model from scratch, a system with a GPU is required. It is recommended that the files be run on a service that provides GPU support such as Google Colab. To train the model on colab, use the following steps:

` !git clone https://github.com/alt-nikitha/Sonnet-Generator.git ` 

` cd Sonnet-Generator ` 

` !pip install -r requirements.txt ` 

` !python pre.py dataset.txt dataset2.txt ` 

` !python train.py ` 

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

Now, we can test the flask application we've built that takes the model we've trained, and generates sonnets when a starting phrase is given to it.

` !python app.py `
    
### Deploying

First, install ngrok library by running
` !pip install flask-ngrok`
ngrok is a lightweight tool used to generate public URLs for testing your model.

Second, make sure in your app.py file, the function app.run(), should not have any debug value.

Third, in your app.py file, import this:-
`from flask_ngrok import run_with_ngrok`

Fourth, in your app.py file, 
type in this command:- 
`run_with_ngrok(app)` right below this 

`app = Flask(__name__,static_url_path='/static')`(Save all these changes in app.py)

Finally, run app.py

`!python app.py`

