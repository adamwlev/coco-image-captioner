# Building the image

docker build -t coco .

# Running the container

docker run -d -p 443:443 -p 80:80 --log-opt max-file 10 coco:latest
