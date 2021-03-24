# this is a building classifier project

### deploy docker image

there are some scripts to build docker image and deploy it.

```shell
domain=chuyuwang
version=0.0.1
image="$domain/building_classifier:$version"
docker build -t $image .
docker push $image
```