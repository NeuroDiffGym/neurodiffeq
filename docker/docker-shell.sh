#!/bin/bash

set -e

# Read the settings file
source ./env.dev

docker build -t $IMAGE_NAME -f Dockerfile ..
docker run --rm --name $IMAGE_NAME -ti --mount type=bind,source="$BASE_DIR",target=/app -p 9898:9898 -e NEURODIFF_API_URL=$NEURODIFF_API_URL $IMAGE_NAME