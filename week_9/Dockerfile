FROM agrigorev/zoomcamp-bees-wasps:v2

RUN pip install Pillow 
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY homework_9.py .
ENV MODEL_NAME=bees-wasps-v2.tflite

CMD [ "homework_9.lambda_handler" ]