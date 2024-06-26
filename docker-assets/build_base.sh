# Run at root of repo!

echo "Building base image..."
sudo docker build -t ghcr.io/biosimulators/bio-check-base .
echo "Built base image."