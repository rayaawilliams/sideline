# Container for Google Cloud Run.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 RAG_PROVIDER=gemini

# CPU-only PyTorch keeps the image much smaller than the default GPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model once, into the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# On each start: pull fresh stats and news, index the news, then serve the app.
# If the NCAA API is down, the app still starts with whatever data the image already has.
CMD (python -m sports.update && python -m rag.ingest --reset) || echo "Update failed; using existing data"; \
    streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
