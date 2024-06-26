#!/usr/bin/env bash

# This script is used to create a sealed secret for the database and jms passwords
# this script should take 5 arguments as input:
#   namespace
#   mongo_user
#   mongo_pswd
#
#   and outputs a sealed secret to stdout
# Example: ./sealed_secret_api.sh remote pswd12345 pswd39393 mongo_user pswd292929 > output.yaml

# validate the number of arguments
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./sealed_secret_api.sh <namespace> <mongo_user> <mongo_pswd>"
    exit 1
fi

SECRET_NAME="api-secrets"
NAMESPACE=$1
MONGO_USERNAME=$2
MONGO_PASSWORD=$3

kubectl create secret generic ${SECRET_NAME} --dry-run=client \
      --from-literal=mongo-username="${MONGO_USERNAME}" \
      --from-literal=mongo-password="${MONGO_PASSWORD}" \
      --namespace="${NAMESPACE}" -o yaml | kubeseal --format yaml
