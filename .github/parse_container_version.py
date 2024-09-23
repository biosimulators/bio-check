import yaml
import sys


service_name = sys.argv[1]

fp = 'docker-compose.yaml'

with open(fp, 'r') as file:
    compose_data = yaml.safe_load(file)

image = compose_data['services'][service_name]['image']
version = image.split(":")[1]

print(version)


