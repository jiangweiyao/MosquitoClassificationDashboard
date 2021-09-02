# Mosquito Classification Dashboard

This repository contains code for
1. Training a mosquito image classification model using PyTorch (retraining Resnet18)
2. Deploying the trained model as an interactive dashboard using Plotly's Dash.
3. Containerization of the above dashboard as a Docker container

## Data
The mosquito images were obtained from Saul Lozano (not included in this repo)

The 6 classes of mosquitoes were Aedes_albopictus, Aedes_vexans, Anopheles_sinensis, Culex_pipiens, Culex_tritaeniorhynchus, Non_vectors.

## Retraining
We retrained the Resnet18 model on the data. We augmented the data using the PyTorch dataloader by randomly flipping the images horizontally and vertically. The final accuracy (from balanced class sets) was 0.542.


## Interactive Dashboard with Plotly's Dash
We deployed the trained Resnet model to predicted the class of the user uploaded image as an interactive dashboard using Plotly's Dash. The user can upload an image using the uploader widget, and the dashboard will classify the image as one of the mosquito classes.


The Python and package dependencies can be cloned from the "dash_environment.yml" using the following command from the Dashboard directory assuming you have Conda installed:
```
conda env create -f dash_environment.yml
```
You can then activate the environment and deploy the code as a Flask server using gunicorn to port 8050 using the command below.
```
conda activate blur_dash
gunicorn -b 0.0.0.0:8050 dash_classifier:server
```

System requirements for dashboard: The dashboard requires 4GB of memory using the trained Resnet model. More cores will speed up the computation. It is currently deployed via Elastic Container Service on a t3.medium (2 vCPUs, 4 GiB) EC2 from AWS.

## Docker container
The dashboard has been containerized using the included Dockerfile in the Dashboard directory. Docker Hub has been configured to automatically build the docker image from this GitHub repository. You can check the link below for the Docker file.
https://hub.docker.com/r/jiangweiyao/mosquitodashboard

You can deploy the Docker container on your computer or VM using the following command using Docker
```
docker run -p 8050:8050 jiangweiyao/mosquitodashboard:latest
```

Alternatively, you can deploy the Docker container as an AWS web service using Elastic Container Services or Fargate. Please follow the instructions from AWS. 
