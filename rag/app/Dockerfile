FROM python:3.11.8-slim-bookworm

WORKDIR /opt/app
ENV HOST=0.0.0.0
ENV LISTEN_PORT 8501
EXPOSE 8501

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends curl bash && \
	rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY main.py rag.py ./
RUN ls -lah /opt/app

CMD ["streamlit", "run", "/opt/app/main.py", "--browser.gatherUsageStats", "false"]
