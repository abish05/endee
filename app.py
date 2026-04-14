import json
import time
from sentence_transformers import SentenceTransformer
import requests 

# Wait, if the python client fails to install, we can fallback to REST.
# I will use the Endee python client!
try:
    from endee import Endee, Precision
    HAS_ENDEE_SDK = True
except ImportError:
    HAS_ENDEE_SDK = False

print("Loading embedding model `all-MiniLM-L6-v2` (dimension 384)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    with open('dataset.json', 'r') as f:
        return json.load(f)

def run_endee_sdk(movies):
    print("Using Endee Python SDK...")
    client = Endee(url="http://localhost:8080")
    
    # Check if index exists, else create it
    index_name = "movies"
    try:
        # Assuming the library throws an error if it doesn't exist 
        client.create_index(name=index_name, dimension=384, space_type="cosine", precision=Precision.FP32)
        print(f"Created index {index_name}")
    except Exception as e:
        print(f"Index might already exist: {e}")
        
    index = client.get_index(index_name)

    vectors_to_upsert = []
    print("Computing embeddings...")
    for movie in movies:
        emb = model.encode(movie['plot']).tolist()
        vectors_to_upsert.append({
            "id": movie['id'],
            "vector": emb,
            "payload": {
                "title": movie['title'],
                "genre": movie['genre'],
                "plot": movie['plot']
            }
        })
    
    print("Upserting vectors into Endee...")
    index.upsert(vectors_to_upsert)
    
    print("\n--- Semantic Search Test ---")
    query = "A movie about space travel and saving the world"
    print(f"Query: '{query}'")
    query_emb = model.encode(query).tolist()
    
    results = index.query(vector=query_emb, top_k=2, include_vectors=False)
    
    for item in results:
        # In case the wrapper returns objects or dicts
        item_id = item.get("id") if isinstance(item, dict) else item.id
        similarity = item.get("similarity") if isinstance(item, dict) else item.similarity
        payload = item.get("payload", {}) if isinstance(item, dict) else item.payload
        print(f"Result -> ID: {item_id}, Similarity: {similarity:.3f}")
        print(f"Title: {payload.get('title')}, Genre: {payload.get('genre')}")
        print(f"Plot: {payload.get('plot')}\n")

def run_endee_rest(movies):
    print("Using Endee REST API fallback...")
    base_url = "http://localhost:8080/api/v1"
    
    try:
        res = requests.post(f"{base_url}/index/create", json={
            "name": "movies",
            "dimension": 384,
            "space_type": "cosine",
            "precision": "FP32"
        })
        print(res.json())
    except Exception as e:
        print("Create Index:", e)

    print("Computing embeddings...")
    vectors_to_upsert = []
    for movie in movies:
        emb = model.encode(movie['plot']).tolist()
        vectors_to_upsert.append({
            "id": movie['id'],
            "vector": emb,
            "payload": {
                "title": movie['title'],
                "genre": movie['genre'],
                "plot": movie['plot']
            }
        })
    
    print("Upserting vectors into Endee...")
    res = requests.post(f"{base_url}/index/movies/upsert", json={"points": vectors_to_upsert})
    print("Upsert Response:", res.json())
    
    print("\n--- Semantic Search Test ---")
    query = "A movie about space travel and saving the world"
    print(f"Query: '{query}'")
    query_emb = model.encode(query).tolist()
    
    res = requests.post(f"{base_url}/index/movies/query", json={
        "vector": query_emb,
        "top_k": 2
    })
    
    if res.status_code == 200:
        results = res.json().get("result", [])
        for item in results:
            print(f"Result -> ID: {item.get('id')}, Similarity: {item.get('similarity', 0):.3f}")
            payload = item.get("payload", {})
            print(f"Title: {payload.get('title')}, Genre: {payload.get('genre')}")
            print(f"Plot: {payload.get('plot')}\n")
    else:
        print("Query failed:", res.text)


if __name__ == "__main__":
    movies = load_data()
    if HAS_ENDEE_SDK:
        run_endee_sdk(movies)
    else:
        run_endee_rest(movies)
