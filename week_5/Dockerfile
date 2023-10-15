# load docker image
FROM svizor/zoomcamp-model:3.10.12-slim

# install pipenv in docker 
RUN pip install pipenv

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# copy the pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install app dependencies from pipfile 
RUN pipenv install --system --deploy

# copy flask app
COPY ["predict_credit_q6.py", "./"]

# expose port 9696 to communicate with flask app 
EXPOSE 9696

# run credit prediction app
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_credit_q6:app"]
