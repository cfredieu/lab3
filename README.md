# Lab 3: Document Retrieval System 
 
Semantic search system using CIFAR-10 training datasets and CNN for ARIN 5360. 

## Quick Start  

```bash
# Install dependencies  
uv sync 

# Start server  
uv run uvicorn src.retrieval.main:app --reload
```

Server starts at url http://localhost:8000  

## Project Structure  
```
lab3/  
|-- src/retrieval/ 
|-- tests/  
|-- static/  
|-- documents/  
|-- pyproject.toml  


