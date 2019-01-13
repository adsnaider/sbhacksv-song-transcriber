import numpy as np
import googleapiclient.discovery

from flask import Flask, jsonify, request, redirect, url_for

import glog as log
log.setLevel('INFO')

app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/')
def send_js():
  return app.send_static_file('html/index.html')

@app.route('/flat-ui.css')
def send_css():
  return app.send_static_file('flat-ui.css')

def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    #GOOGLE_APPLICATION_CREDENTIALS="song-transcriber-1ca56987a10e.json"
    
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

project = "song-transcriber"
model = "song_convert"
version = "Zero"

@app.route('/handle_data', methods=['POST', 'GET'])
def handle_data():
  if request.method == 'POST':
    log.debug(request.form)
    msg = request.form['MSG']
    #convert to floats 
    array = [[float(x) for x in msg.split()]]
    instances = {'model_input': array}
    response = predict_json(project, model, instances, version)
    return str(response[0]['model_output'])

if __name__ == '__main__':
  app.run(port=9876)