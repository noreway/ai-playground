
# Vorbereitungen
- Docker Desktop installieren https://www.docker.com/products/docker-desktop/
- OpenAI Konto erstellen und API Key generieren https://platform.openai.com/ (optional)
- `.env.template` zu `.env` kopieren und Konfiguration anpassen (für LangSmith, OpenAI, Atlassian und Azure)

# Starten
```bash
# CPU only
docker compose up --build
```

# Benutze Sprachmodelle laden
```bash
winpty docker exec -it ai-playground-open-webui_ollama ollama pull mistral
winpty docker exec -it ai-playground-open-webui_ollama ollama pull llama2
```

# App öffnen
- http://localhost:3000

# Links
Setup
- https://docs.openwebui.com/
- https://github.com/open-webui/open-webui/blob/main/docker-compose.yaml
Features TODO
 - setup RAG and local Chroma
   https://docs.openwebui.com/getting-started/env-configuration#rag
 - Speech to Text and Text to Speech
   https://docs.openwebui.com/getting-started/env-configuration#speech-to-text
 - Image generation
   https://docs.openwebui.com/getting-started/env-configuration#image-generation
