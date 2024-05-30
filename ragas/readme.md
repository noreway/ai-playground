
# Vorbereitungen
- Docker Desktop installieren https://www.docker.com/products/docker-desktop/
- OpenAI Konto erstellen und API Key generieren https://platform.openai.com/ (optional)
- `.env.template` zu `.env` kopieren und Konfiguration anpassen (für OpenAI)

# Starten
```bash
# CPU only
docker compose up --build
# with GPU
docker compose --file docker-compose_gpu.yml up --build
```

# Links
- https://docs.ragas.io/en/stable/howtos/integrations/langchain.html
