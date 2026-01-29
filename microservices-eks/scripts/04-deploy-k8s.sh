#!/usr/bin/env bash
set -euo pipefail

cd terraform
ECR_API="$(terraform output -raw ecr_api)"
ECR_AUTH="$(terraform output -raw ecr_auth)"
ECR_WEB="$(terraform output -raw ecr_web)"
cd ..

TAG="1.0.0"

# Replace image placeholders
sed -i.bak "s|REPLACE_ME_ECR_API:1.0.0|${ECR_API}:${TAG}|g" k8s/10-api.yaml
sed -i.bak "s|REPLACE_ME_ECR_AUTH:1.0.0|${ECR_AUTH}:${TAG}|g" k8s/11-auth.yaml
sed -i.bak "s|REPLACE_ME_ECR_WEB:1.0.0|${ECR_WEB}:${TAG}|g" k8s/12-web.yaml

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/10-api.yaml
kubectl apply -f k8s/11-auth.yaml
kubectl apply -f k8s/12-web.yaml
kubectl apply -f k8s/20-hpa-api.yaml
kubectl apply -f k8s/21-hpa-auth.yaml
kubectl apply -f k8s/30-ingress-alb.yaml

echo ""
echo "Waiting for Ingress ALB..."
kubectl -n microservices get ingress microservices-ingress -w
