In week 9️⃣ of the ML Zoomcamp we covered:

☁️🚀  Serverless Deployment  
We discuss how to deploy models using AWS Lambda, which allows us to run our service without managing any infrastructure. We refer to this setup as serverless deployments. Another advantage of serverless deployments is that we only pay for what we use rather than what we rent. We explore the AWS Lambda environment and learn how to define and test a lambda function.  

🎈Tensorflow lite  
We learn how to convert a trained keras model to tensorflow lite (TFL). TFL is a much smaller library than tensorflow, however, it can only be used for inference. To create a tflite model, we first need to define a converter using `tf.lite.TFLiteConverter.from_keras_model(model)` command, then use it to load a tflite version of the model. 

✈️ Using tflite Models for Inference  
To use a tflite model:  
1- We create an interpreter using the `tflite.Interpreter(model_path='model.tflite')` command  
2-Define the input and output indices:  
```
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']  
```  
3- Finally, we set the input, invoke the interpreter, and call get the predictions using these commands: 
```
interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
```  
🌐💻Deploying the code to Lambda  
To deploy our code to AWS Lambda, first, we need to create a docker image for the code making sure that it has a function called `lambda_handler()` that follows the lambda function structure. We then import the docker image to the lambda api using the ECR command line interface and create a `REMOTE_URI`. Then, we tag the docker image with the created uri and push it. Next, we define the lambda settings, such as timeout and ram, from the control panel. Finally, we test and run the function.   

📚 Our homework involved:  
* Converting a keras model to tflite
* Defining the input and output indices for the tflit model
* Preprocessing an image to be used by the model
* Predicting the class of the image
* Extending a docker image with a different model version of the model, and predicting the class of the image using it.  

👉 The code for this project can be found [here](https://github.com/el-grudge/mleng-zoomcamp/tree/main/week_9).

#mlzoomcamp #ml_engineering #data_science #learning_in_public #boosting #decision_trees
