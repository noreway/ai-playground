
# Vorbereitungen
- Docker Desktop installieren https://www.docker.com/products/docker-desktop/
- LangSmith Konto erstellen und API Key generieren https://smith.langchain.com/ (optional)
- OpenAI Konto erstellen und API Key generieren https://platform.openai.com/ (optional)
- Atlassian API Key generieren (optional)
- Azure AI Search Api Key generieren (optional)
- `.env.template` zu `.env` kopieren und Konfiguration anpassen (für LangSmith, OpenAI, Atlassian und Azure)

# Starten
```bash
# CPU only
docker compose up --build
# with GPU
docker compose --file docker-compose_gpu.yml up --build
```

# Benutze Sprachmodelle laden
```bash
winpty docker exec -it ai-playground-rag_ollama ollama pull mistral
winpty docker exec -it ai-playground-rag_ollama ollama pull llama2
```

# App öffnen
- http://localhost:8501
- In der Sidebar Sprachmodell auswählen (gpt-x Varianten sind nur verfügbar wenn OpenAI API Key definiert wurde)
- Eine Knowledge Base anlegen und Dokumente hochladen
- Fragen...

# Links
Gutes Tutorial:
- https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895
  -> kurz: https://research.ibm.com/blog/retrieval-augmented-generation-RAG
  -> lang: https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1

LangChain:
- https://nanonets.com/blog/langchain/
- https://www.infoworld.com/article/3705035/5-easy-ways-to-run-an-llm-locally.html?page=2

Ollama Beispiele:
- https://python.langchain.com/docs/integrations/llms/ollama
- https://github.com/ollama/ollama/tree/main/examples

Ollama in Docker ausführen:
- https://khandelwal-shekhar.medium.com/deploy-ollama-using-official-docker-image-c09a27cd332f
- https://github.com/valiantlynx/ollama-docker/blob/main/src/rag.py

ChromaDB in separatem Docker Container ausführen:
- https://abhishektatachar.medium.com/run-chroma-db-on-a-local-machine-and-as-a-docker-container-a9d4b91d2a97

Confluence Loader:
- https://python.langchain.com/docs/integrations/document_loaders/confluence
- https://medium.com/@jeffgeiser/confluence-and-langchain-735c67db193a
- https://medium.com/@ahmed.mohiuddin.architecture/ai-powered-confluence-search-using-langchain-azure-openai-and-azure-cognitive-search-f9765c625b70
- https://www.shakudo.io/blog/building-confluence-kb-qanda-app-langchain-chatgpt

Mistral Prompt:
- https://www.promptingguide.ai/models/mistral-7b#capabilities

Azure AI Search:
 - https://medium.com/@ahmed.mohiuddin.architecture/ai-powered-confluence-search-using-langchain-azure-openai-and-azure-cognitive-search-f9765c625b70
 
Unterschied OpenAI vs. Azure OpenAI:
 - https://www.uscloud.com/blog/the-differences-between-openai-and-microsoft-azure-openai/#:~:text=Your%20data%20is%20your%20own,grow%20using%20advanced%20artificial%20intelligence

Vector database embedding functions:
 - https://www.heise.de/hintergrund/Sinnvoll-geclustert-Smarte-Suche-mit-Vektoren-9639467.html
 - https://www.heise.de/hintergrund/Sorgfaeltiger-als-ChatGPT-Embeddings-mit-Azure-OpenAI-Qdrant-und-Rust-9066795.html
 - https://huggingface.co/datasets/mikeee/chroma-paraphrase-multilingual-mpnet-base-v2