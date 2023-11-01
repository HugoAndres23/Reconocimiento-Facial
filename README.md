# Face Recognition API

## Tools

- Face Recognition model: [deepface](https://github.com/serengil/deepface)

- API: [fastapi](https://github.com/tiangolo/fastapi)

## Docker

Directory structure in docker container:

```
├── /app
│   ├── app
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── utils.py
|   |
│   ├── data  # Database - Folder of images
│   │   ├── *.[jpg | jpeg]
|   |
│   ├── query  # Query - Folder of images need to find id
│   │   ├── *.[jpg | jpeg]
|   |
```

1. Correr Docker

```
docker compose up --build
```

2. Abrir: localhost:80
