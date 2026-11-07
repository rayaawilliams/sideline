# Documents for search

`python -m sports.update` saves NCAA news articles to `data/docs/news/`. You can also add your own
files here (PDF, .txt or .md), such as notes on your favourite teams.

After adding or changing files, rebuild the search index:

```bash
python -m rag.ingest --reset
```
