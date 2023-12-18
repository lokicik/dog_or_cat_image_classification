import json
import requests

url = 'https://dog-or-cat.onrender.com/lambda_function'


#################################################################################################################
# CHANGE THE LINK BELOW TO EVALUATE DIFFERENT IMAGES WITH THE MODEL
link = "https://d.newsweek.com/en/full/1809693/cat-dog.webp?w=1600&h=900&q=88&f=772f894f001bafc6c5094cc33d71bc19"
#################################################################################################################
payload = {'url': link}

response = requests.post(url, json=payload).json()
pretty_response = json.dumps(response, indent=4)

print(pretty_response)
