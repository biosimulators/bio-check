#!/usr/bin/env bash
# When to use this script: IF CHANGES TO SECRETS HAVE BEEN MADE, run the dev_secrets script to regenerate (ie: Mongo Username, mongo password, and/or mongodb connection uri)


current_dir=$(pwd)
deployment="$1"  # ie: dev, prod, etc
_wait="$2"  # -w : whether to wait for the tunnel process to complete. Defaults to nothing.

set -e

if [ -z "$deployment" ]; then
  echo "You must pass one of these three values as a runtime argument for environment to which changes will be applied: dev, prod, test"
  exit 1
fi


function create_sealed_secrets {
  local origin=$1

  # go to ssh lib and run script for secrets
  cd ~/.ssh || { echo "Failed to change directory to ~/.ssh"; exit 1; }
  ./dev_secrets.sh
  echo "Successfully ran secrets script."

  # Return to the original directory
  cd "$origin" || { echo "Failed to change directory to $origin"; exit 1; }
}

function apply_overlays {
  deployment=$1  # ie: dev, prod, test
  kubectl kustomize overlays/$deployment | kubectl apply -f -
  echo "overlays/${deployment} successfully applied!"
}

function wipe_overlays {
  deployment=$1  # ie: dev, prod, test
  kubectl kustomize overlays/$deployment | kubectl delete -f -
  echo "overlays/${deployment} successfully wiped!"
}

function refresh_overlays {
  deployment=$1
  wipe_overlays "$deployment"
  apply_overlays "$deployment"
}


# 1. start mini kube
minikube start --base-image gcr.io/k8s-minikube/kicbase-builds:v0.0.42-1703092832-17830 --driver docker  --memory=32g --cpus=8

# 2. log into grafana
kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo

# 3. set up ingress
kubectl get pods -n ingress-nginx

# 4. create secret
kubectl create secret generic secret-name --dry-run=client --from-literal=foo=bar -o yaml | \
    kubeseal \
      --controller-name=sealed-secrets-controller \
      --controller-namespace=kube-system \
      --format yaml > mysealedsecret.yaml

# 5. seal secret
kubectl apply -f mysealedsecret.yaml

# 6. check cert manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml \
  && cmctl check api

# 7. re-establish DNS config
echo "127.0.0.1 minikube.local" | sudo tee -a /etc/hosts

# 8. verify kustomization scripts
kubectl kustomize overlays/dev | kubectl apply --dry-run=client --validate=true -f -

# 9. apply kustomization scripts (this will usually overrwrite existing content)
# apply_overlays "$deployment"

# 9a. Optionally, you may choose (or are required) to wipe scripts to start fresh. In that case, first wipe and then re-apply
refresh_overlays "$deployment"

# 10. Re-create/update sealed secrets!
create_sealed_secrets "$current_dir"

# 11a. OPEN A NEW TERMINAL WINDOW! Ensure minikube is running IN NEW WINDOW:
# open a new window!
./start-tunnel.sh &

TUNNEL_PID=$!

# 11b. Apply dev overlays:
apply_overlays "$deployment"

# 11c. Optionally wait for process to complete
# wait $TUNNEL_PID