# ML_production_IZ2

## _Loading docker inference image_

This part of homework and this branch of repository is dedicated for development of online inference of model, trained in previous homework. Solution to this task is docker image of uvicorn server, hosted on docker hub. In order to run the server on your local machine, one need to do next steps:
```
docker pull gotovtsev/ml_prod_2
docker run -dp 8000:8000 --rm gotovtsev/ml_prod_2
```

Parametrizing server inside docker container is done by environment variables, for example, to load model artifact from inside docker image, execute next command:

```
docker run -dp 8000:8000 --rm --env MODEL_LOCATION=local [--env PATH_TO_SERIALIZATION=your_path] gotovtsev/ml_prod_2
```
```PATH_TO_SERIALIZATION``` can be ommited because it has default value.

For loading another artifact of model from google disk, execute next command:
```
docker run -dp 8000:8000 --rm --env PATH_TO_SERIALIZATION=<your_path> gotovtsev/ml_prod_2
```
,where ```<your_path>``` is sha of shareble link from google disk.

For checking that server is functioning, in repository located simple script named ```test_request.py```, run it after the docker container with server is up. In output there will be clear trace.