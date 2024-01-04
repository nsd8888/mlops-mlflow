
# serve: deploys the model as a local REST API server.

# build_docker: packages a REST API endpoint serving the model as a docker image.

# predict: uses the model to generate a prediction for a local CSV or JSON file. Note that this method only supports DataFrame input.


import requests

inference_request = {
        "dataframe_records": [[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]]
}
endpoint = "http://localhost:1234/invocations"
response = requests.post(endpoint, json=inference_request)
print(response.text)