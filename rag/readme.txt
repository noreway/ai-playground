
# prerequisits
# install docker desktop
# create a Langsmith account, optional
# create and OpenAI account, optional
# copy rename .env.template to .env and add required settings

# build and run
docker compose up --build

# pull models
winpty docker exec -it ai-playground-rag_ollama ollama pull mistral
winpty docker exec -it ai-playground-rag_ollama ollama pull llama2


used tutorial:
https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895

langchain guide:
https://nanonets.com/blog/langchain/
https://www.infoworld.com/article/3705035/5-easy-ways-to-run-an-llm-locally.html?page=2

ollama examples:
https://python.langchain.com/docs/integrations/llms/ollama
https://github.com/ollama/ollama/tree/main/examples

run ollama in docker:
https://khandelwal-shekhar.medium.com/deploy-ollama-using-official-docker-image-c09a27cd332f
https://github.com/valiantlynx/ollama-docker/blob/main/src/rag.py

chroma in separate docker container:
https://abhishektatachar.medium.com/run-chroma-db-on-a-local-machine-and-as-a-docker-container-a9d4b91d2a97

confluence loader:
https://python.langchain.com/docs/integrations/document_loaders/confluence
https://medium.com/@jeffgeiser/confluence-and-langchain-735c67db193a
https://medium.com/@ahmed.mohiuddin.architecture/ai-powered-confluence-search-using-langchain-azure-openai-and-azure-cognitive-search-f9765c625b70
https://www.shakudo.io/blog/building-confluence-kb-qanda-app-langchain-chatgpt

mistral prompt:
https://www.promptingguide.ai/models/mistral-7b#capabilities

erkenntnisse:
 - grosse llm wie gpt4 (extern, daten, rechtliches) vs. lokales hosting (kleine modelle, langsam, hoher leistungsbedarf)
 - wir können kein eigenes llm trainieren und müssen bestehende benutzen, diese kennen die inhalte der kunden nicht
   RAG wird für uns eine wichtige architektur sein
 - als sprache werden wir wohl python benutzen müssen um vielseitige und aktuelle auswahl an tools zu haben
   das lokale hosting ist bereits einschränkung genug
 - app kann trotzdem auf blueprint aufbauen, nur die RAG chain wird mit python gebaut und bietet dann eine REST API für QA
 - wir haben viel zu lernen, verständnis ist hier wichtig, auch wenn wir die "tools" nur benutzen
 - daher schnelles prototyping mit docker compose (hosting), ollama (llm server), langchain (rag framework), chroma (vector db) und streamlit (proto app mit ui) hilfreich
 - Debugging mit LangSmith sehr hilfreich (Problem: warum macht er mir keine Referenzen mit den originalen Dateinamen)
 - Prompt Engineering ist wichtig und schwierig!
 - LLM wechseln bedeutet auch Prompt wehseln und neu tüfteln, messen, tunen
 - OpenAI GPT ist deutlich schnellen als Ollama lokal, unklar wie viel macht das Model und wie viel das Hosting, bzw. die Hardware aus. 
 