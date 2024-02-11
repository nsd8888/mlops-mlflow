
# serve: deploys the model as a local REST API server.

# build_docker: packages a REST API endpoint serving the model as a docker image.

# predict: uses the model to generate a prediction for a local CSV or JSON file. Note that this method only supports DataFrame input.


import pickle
import requests

with open("Std.pkl", "rb") as f:
        std = pickle.load(f)



inference_request = {
        "dataframe_records": [[23,"F","HIGH","HIGH",25.355]]
}
endpoint = "http://localhost:5000/predict"
response = requests.post(endpoint, json=inference_request)
print(response.text)