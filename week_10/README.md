In week 1️⃣0️⃣ of the ML Zoomcamp we covered:

🚀 TF Serving
We learn about TensorFlow Serving, another component in the TensorFlow ecosystem. Like TF-Lite, TF Serving models can be used for inference only. However, TF-Lite and TF Serving have different purposes, the former is mainly used for deployment on edge devices, the latter is designed for serving ML models on production ready environments, such as cloud servers. To create a TF Serving, a model needs to be saved in TF format using `tf.saved_model.save(model, '<target-name>')`, which will create a directory with a number of files and sub-directories. We can use the `saved_model_cli` utility to navigate the model different components, such as its inputs and outputs. 

🪄 Docker-compose
We explore how to use the docker-compose tool to setup and run multi-container applications. In our example, we had two containers, the gateway container which handles input preprocesses and the model container. Docker-compose uses a yaml configuration file to define the enviornment that includes both containers and how they communicate with each other. After defining the yaml file, we use the `docker-compose up` to start the app.

🕹️🕸️ Kubernetes 
We learn about kubernetes, a technology that facilitates the deployment, scaling and operation of containers. It is a more powerful orchestration tool than docker-compose, and is more suitable for large-scale deployments. The main kubernetes component is the pod, which is an abstraction of a docker container. Pods exist inside of nodes and are grouped in deployments across different nodes. We use yaml configuration files to define kubernetes deployments. A common deployment.yaml file has the following sections:

* metadata: contains the app name and label 
* specs: defines the deployment state, including the number of replicas and pod template 

We also learn about kubernetes services and create one to define an access point for our deployment. Kubernetes services are also defined using yaml configuration files.

📚 Our homework involved:  

* Deploying a docker image
* Installing kubectl and kind
* Creating a kubernetes cluster
* Registering a docker image with kind
* Defining a kubernetes `deployment.yaml`
* Defining a kubernetes `service.yaml`

👉 The code for this project can be found [here](https://github.com/el-grudge/mleng-zoomcamp/tree/main/week_10).

#mlzoomcamp #ml_engineering #data_science #learning_in_public #kubernetes #docker