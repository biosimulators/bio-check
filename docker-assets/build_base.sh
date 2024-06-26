# Run at root of repo!

prune="$1"  # -p
if [ "$prune" ]; then
  docker system prune -a -f
fi

echo "Building base image..."
sudo docker build -t ghcr.io/biosimulators/bio-check-base .
echo "Built base image."
