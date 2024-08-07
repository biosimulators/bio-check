# local minikube config


### Apply biochecknet overlays (commonly used):

```bash
cd kustomize \
  && kubectl kustomize overlays/biochecknet | kubectl apply -f - \
  && cd ..
```


0. Build base 0.0.2
0a. Check gcloud creds in base
0b. Push base 0.0.2
1. build api 0.0.0
2. test gcloud creds 
3. push api 0.0.0
4. build worker 0.0.0
5. test gcloud creds
6. push worker 0.0.0
7. Perform readme wf again


## ArgoCD setup

according to https://argo-cd.readthedocs.io/en/stable/getting_started/

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

make the following a table

| technology                             | description                                                            |
|----------------------------------------|------------------------------------------------------------------------|
| Kubespray (on prem cluster)            | ArgoCD (GitOps), Sealed Secrets, Certificate Manager         |
| Lens                                   | nice visual tool for Kubernetes clusters |
| minikube  (local dev cluster)          | kubectl (manual deploy), plain secrets, self-signed certs |
| Kustomize                              | to organize k8s manifests for multiple environments                    |
| ArgoCD                                 | for continuous deployment and GitOps                                   |
| Sealed Secrets                         | for secret management of encrypted secrets in Git per each cluster     |
| Certificate Manager with Let's Encrypt | for automatic refresh of SSL certificates                              |
| Ingress controller                     | for reverse proxies and CORS handling                                  |
| Persistent Volumes/Claims              | to map NFS mounts to pods                                              |

# local minikube config

### install Lens

### install and start minikube on macos

```bash
brew install qemu
brew install socket_vmnet
brew tap homebrew/services
HOMEBREW=$(which brew) && sudo ${HOMEBREW} services start socket_vmnet
# minikube start --driver qemu --network socket_vmnet --memory=8g --cpus=2
minikube start --base-image gcr.io/k8s-minikube/kicbase-builds:v0.0.42-1703092832-17830 --driver docker  --memory=32g --cpus=8
minikube addons enable metrics-server

brew install kubectl
brew install helm
```

### install kube-prometheus-stack

see https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack
`
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

kubectl create namespace monitoring
helm install prometheus --namespace monitoring prometheus-community/kube-prometheus-stack
```

in Lens, you can see the prometheus pods and services in the monitoring namespace.  
Log into Grafana with admin and the password from the following command.

```bash
kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
```

### set up ingress controller

```bash
minikube addons enable ingress
kubectl get pods -n ingress-nginx
```

### Sealed Secrets setup

install sealed secrets and the controller

```bash
brew install kubeseal
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm install sealed-secrets -n kube-system \
     --set-string fullnameOverride=sealed-secrets-controller sealed-secrets/sealed-secrets
```

create a sealed secret:

```bash
kubectl create secret generic secret-name --dry-run=client --from-literal=foo=bar -o yaml | \
    kubeseal \
      --controller-name=sealed-secrets-controller \
      --controller-namespace=kube-system \
      --format yaml > mysealedsecret.yaml
```

**If necessary, you may need to create the secret using the public cert of your sealed-secrets installation as such:**

```bash
kubectl create secret generic secret-name --dry-run=client --from-literal=foo=bar -o yaml | \
    kubeseal \
      --controller-name=sealed-secrets-controller \
      --controller-namespace=kube-system \
      --format yaml > mysealedsecret.yaml \
      --cert=/path/to/the/sealed/secrets/cert 
```

apply the sealed secret:

```bash
kubectl apply -f mysealedsecret.yaml
```

### ArgoCD setup

### Certificate Manager setup

```bash
brew install cmctl
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml
cmctl check api
```

# Configure minikube networking for local development

### set up DNS entries for ingress rounting

* vcell-api, vcell-rest, and s3proxy services are mapped to minikube.local

* vcell-webapp is mapped to minikube.local

* create local DNS entries for minikube.local and webapp.minikube.local

```bash
#echo "$(minikube ip) minikube.local" | sudo tee -a /etc/hosts
echo "127.0.0.1 minikube.local" | sudo tee -a /etc/hosts
```

**FOR PROD:**

* **note** on mapping to localhost rather than minikube ip address:  
   from https://github.com/kubernetes/minikube/issues/13510.  "Hi, I can confirm that running minikube tunnel works for me on m1 with the docker driver.
   Keep in mind that your etc/hosts file needs to map to 127.0.0.1, instead of the output
   of minikube ip or kubectl get ingress - this is an important gotcha."

# deploying the vcell services to minikube

### verify the kustomization scripts

```bash
kubectl create namespace dev
kubectl kustomize overlays/dev | kubectl apply --dry-run=client --validate=true -f -
```

### apply the kustomization scripts

```bash
kubectl kustomize overlays/dev | kubectl apply -f -
```

### Wipe 
```bash
kubectl kustomize overlays/dev | kubectl delete -f -
```

### create sealed secrets (see [scripts/README.md](scripts/README.md))

# expose services from minikube cluster

### expose ingress routing to localhost as minikube.local and webapp.minikube.local

for vcell-rest, vcell-api and s3proxy services

1. Ensure minikube is running:
```bash
sudo minikube tunnel
```

2. Apply dev overlays:
```bash
kubectl kustomize overlays/dev | kubectl apply -f -
```

2a. Apply **biocheck** overlays:
```bash
kubectl kustomize overlays/biochecknet | kubectl apply -f -
```

### expose JMS and Mongo services to UCH routable ip address

for activemqsim service to receive status messages from simulation workers on HPC cluster

```bash
export EXTERNAL_IP=$(ifconfig | grep 155.37 | awk '{print $2}' | cut -d'-' -f1)
export DEV_NAMESPACE=remote
# bypass services of type LoadBalancer or NodePort - directly export deployment ports
sudo kubectl port-forward --address ${EXTERNAL_IP} -n ${DEV_NAMESPACE} deployment/activemqsim 8161:8161
sudo kubectl port-forward --address ${EXTERNAL_IP} -n ${DEV_NAMESPACE} deployment/activemqsim 61616:61616
sudo kubectl port-forward --address ${EXTERNAL_IP} -n ${DEV_NAMESPACE} deployment/mongodb 27017:27017
# set jmshost_sim_external to $EXTERNAL_IP in ./config/jimdev/submit.env
sed -i '' "s/jmshost_sim_external=.*/jmshost_sim_external=${EXTERNAL_IP}/" ./config/jimdev/submit.env
```

# running the VCell Client

run VCell Java Client (cbit.vcell.client.VCellClientMain) against local minikube

1. set VM Option flags to tolerate the self-signed cert

```sh
-Dvcell.ssl.ignoreHostMismatch=true
-Dvcell.ssl.ignoreCertProblems=true
```

2. use local DNS entry for minikube cluster (see spec.tls.hosts in /overlays/devjim/vcell-ingress.yaml)

```sh
--api-host=minikube.local:443
```

# debugging

1. lightweight local log tailing with logtail

```bash
brew tap johanhaleby/kubetail
brew install kubetail
export deployment=<THE NAME OF YOUR DEPLOYMENT>  # or any deployment (prod, test, remote)
kubetail -n "$deployment"  
```