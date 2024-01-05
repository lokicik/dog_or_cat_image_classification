# This file was actually for AWS but I couldn't do it, the actual predict.py is lambda_function.py

import requests

def invoke(image_url):
    # deploy by running docker image locally on port 9000
    lambda_invoke_url = 'http://0.0.0.0:8080/2015-03-31/functions/function/invocations'
    # publish docker image to aws ecr, creating aws lambda function & exposing it via aws API gateway
    # lambda_invoke_url = ''
    payload = {'url': image_url}

    response = requests.post(lambda_invoke_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    test_image_url = 'https://www.shutterstock.com/image-photo/raw-my-cats-picture-260nw-1325927822.jpg'
    result = invoke(test_image_url)
    print('image')
    print(result)
    print()
