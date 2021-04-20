#!/bin/bash

set -e

# Read the settings file
source ./env.dev

# Create the network if we don't have it yet
docker network inspect neurodiffhub >/dev/null 2>&1 || docker network create neurodiffhub

docker build -t $IMAGE_NAME -f Dockerfile ..
docker run --rm --name $IMAGE_NAME -ti --mount type=bind,source="$BASE_DIR",target=/app -p 9898:9898 -e NEURODIFF_API_URL=$NEURODIFF_API_URL --network neurodiffhub $IMAGE_NAME