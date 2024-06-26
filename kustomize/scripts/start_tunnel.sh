#!/usr/bin/env bash

status=$(minikube status | grep "Running")
if [ -n "$status" ]; then
    minikube stop
fi

minikube start --base-image gcr.io/k8s-minikube/kicbase-builds:v0.0.42-1703092832-17830 --driver docker  --memory=32g --cpus=8
minikube addons enable metrics-server

# start prometheus monitoring
if helm install prometheus --namespace monitoring prometheus-community/kube-prometheus-stack; then
    echo "Prometheus is set up!"
else 
    echo "Prometheus still in use"
fi 


# start minikube tunnel
sudo minikube tunnel
