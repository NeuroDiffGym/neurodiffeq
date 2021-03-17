call env.bat
docker build -t %IMAGE_NAME% -f Dockerfile ..
cd ..
docker run --rm --name %IMAGE_NAME% -ti --mount type=bind,source="%cd%",target=/app -p 9898:9898 %IMAGE_NAME%

