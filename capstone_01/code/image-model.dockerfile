FROM tensorflow/serving:2.7.0

COPY ./location_classifier /models/location_classifier/1 
ENV MODEL_NAME="location_classifier"