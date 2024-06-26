# Run at root of repo!

prune="$1"  # -p
if [ "$prune" ]; then
  docker system prune -a -f
fi

echo "Building base image..."
docker build -f ./Dockerfile-base -t ghcr.io/biosimulators/bio-check-base .
echo "Built base image."
