#!/usr/bin/env bash

set -e 

function start_minikube {
    if minikube start --base-image gcr.io/k8s-minikube/kicbase-builds:v0.0.42-1703092832-17830 --driver docker  --memory=32g --cpus=8; then
        echo "Minikube successfully started!"
    fi 
}

function start_tunnel {
    # start minikube
    if minikube status | grep Stopped; then 
        start_minikube 
    else 
        echo "Minikube has already started."
    fi 

    # start prometheus monitoring
    helm install prometheus --namespace monitoring prometheus-community/kube-prometheus-stack

    # start minikube tunnel
    sudo minikube tunnel
}

start_tunnel

# to stop: minikube stop
