# Use the official Lambda Python runtime
FROM python:3.8.12-slim

# Install required dependencies
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl
RUN pip install keras tensorflow waitress
RUN pip install urllib3==1.26.7 pillow docker
RUN pip install flask
# Install dependencies




# Create a directory for your Lambda function
WORKDIR /app

# Copy necessary files
COPY dog_cat_classifier.tflite .
COPY lambda_function.py .
COPY template.yaml .
# Set environment variable for the model name
ENV MODEL_NAME=dog_cat_classifier.tflite



EXPOSE 9000
EXPOSE 8080

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8080", "lambda_function:app"]
