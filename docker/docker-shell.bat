call env.bat
docker build -t %IMAGE_NAME% -f Dockerfile_win ..
cd ..
docker run --rm --name %IMAGE_NAME% -ti --mount type=bind,source="%cd%",target=/app %IMAGE_NAME%

