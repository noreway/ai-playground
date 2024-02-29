
# build image
docker build -t ai-playground/ollama .

# run container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ai-playground-ollama ai-playground/ollama

# work with models
winpty docker exec -it ai-playground-ollama bash

winpty docker exec -it ai-playground-ollama ollama pull llama2
winpty docker exec -it ai-playground-ollama ollama run llama2

winpty docker exec -it ai-playground-ollama ollama create mario -f ./mario.Modelfile
winpty docker exec -it ai-playground-ollama ollama run mario

winpty docker exec -it ai-playground-ollama ollama create coach -f ./coach.Modelfile
winpty docker exec -it ai-playground-ollama ollama run coach
