---
title: WeMakeDev RAG API  
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: mit
app_port: 7860
disable_embedding: false
suggested_hardware: t4-small
---

# WeMakeDev RAG API

A Retrieval-Augmented Generation (RAG) API that automatically downloads and processes educational books from Hugging Face for intelligent question answering.

## 🚀 Quick Start

1. The API will automatically download books from `Navanihk/books` repository
2. Make a POST request to `/pdf` with your query
3. Get intelligent answers based on the book content

## 📚 Example Usage

```bash
curl -X POST "https://your-space-url/pdf" \
-H "Content-Type: application/json" \
-d '{"query": "What is machine learning?"}'
```

## 🔗 Endpoints

- `GET /` - API information
- `GET /docs` - Interactive API documentation  
- `GET /books` - List available books
- `POST /pdf` - Query the books