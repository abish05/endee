# AI Semantic Search with Endee

This repository contains an AI/ML project that demonstrates a **Semantic Search** application using the [Endee](https://github.com/endee-io/endee) vector database.

## Project Overview

The project creates a **Movie Recommendation System** using semantic search. It uses a small dataset of movie plots and matches a user's free-text natural language query (like "A movie about space travel") against the plots to return the most relevant movies. This is a foundational step for building Retrieval-Augmented Generation (RAG) and Agentic AI workflows.

## System Design

1. **Dataset**: A JSON dataset (`dataset.json`) containing movies with their IDs, titles, genres, and textual plots.
2. **Embedding Model**: We use the `sentence-transformers` open-source model (`all-MiniLM-L6-v2`) to convert movie plots into 384-dimensional dense vectors.
3. **Vector Database**: We use **Endee**, a high-performance open-source vector database, to store these dense vectors along with the metadata payload (title, genre, plot).
4. **Retrieval**: When a query is given, it is converted to a vector using the same model. We perform a cosine similarity search against Endee to fetch the most similar matches.

## Use of Endee

Endee is used as the core retrieval engine. The application utilizes the Endee Python SDK (or HTTP REST API equivalent) to:
- Connect to the local Endee server.
- Create a collection index named `movies` with dimension `384` and `cosine` distance metric.
- Upsert dense vectors and their associated `payload` (JSON metadata).
- Execute `query` commands to retrieve the top matches based on semantic similarity.

## Setup Instructions

### Prerequisites
1. **Docker**: Ensure Docker is installed to run Endee.
2. **Python 3.8+**: Ensure you have Python installed.

### 1. Run Endee Database
First, start the Endee server locally using Docker:

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

Wait a few seconds for the server to start on `http://localhost:8080`.

### 2. Install Project Dependencies
In a new terminal window, navigate to this project folder.
It is highly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

*(Note: If `endee` Python SDK is available, it will be installed. Otherwise, the script acts via REST fallback.)*

### 3. Run the Application
Start the semantic search application:

```bash
python app.py
```

### Expected Output
The script will output the loading of the model, upsert the movie dataset into the Endee database, and perform a test query: **"A movie about space travel and saving the world"**.

You should see matching movies (like *Interstellar*) successfully retrieved by the API based purely on semantic text similarity of their plots!
