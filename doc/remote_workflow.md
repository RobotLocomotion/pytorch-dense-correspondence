## Remote workflow

Here are some tips on remote working on this repo over ssh / etc.

#### 1. Ssh in and access already-running container
If you already have a container running on a machine, but want to ssh in, you can still enter the already-running container:

```
docker exec -it pytorch-container bash
```

Where `pytorch-container` is the name of the container as for example displayed form `docker ps`.

#### 2. Find already-running notebook tokens

If you have a notebook running inside that container, once in the container you can figure out the notebook's token by:

```
jupyter notebook list
```

Which will for example give you:

```
Currently running servers:
http://0.0.0.0:8888/?token=a54c73014d7bcd57ea8f6c8e4f0803fb6d97d338d932de87 :: /home/peteflo/code
```

#### 3. Jump into already-running notebook session 
You can then hop into your previously-started notebook session by entering in the IP in your remote browser:

```
http://replace.with.actual.ip:8888/?token=a54c73014d7bcd57ea8f6c8e4f0803fb6d97d338d932de87
```
