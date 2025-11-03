# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import shutil
import os

# Import your core logic functions
from basic import query_pinecone_for_answer, upload_pdf_to_pinecone

app = FastAPI()

origins = [
    "http://localhost:3000",      
    "http://localhost:5173",      
    "http://127.0.0.1:5173",     
]


    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins temporarily for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    query: str
    mode: str  # 'owner' or 'mechanic'



@app.post("/api/ask", response_model=Dict[str, Any])
async def ask_question(request: QueryRequest):
    """Endpoint for the chatbot query."""
    try:
        answer = query_pinecone_for_answer(request.query, request.mode)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.post("/api/upload")
async def upload_manual(file: UploadFile = File(...)):
    """Endpoint to upload a PDF and process it into Pinecone."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Read file content
        content = await file.read()
        
        # Process and upload the file using the core logic
        result = upload_pdf_to_pinecone(content, file.filename)
        
        return {"message": "PDF processed and uploaded successfully.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload and processing failed: {e}")


ASSET_DIR = "pdf_assets"
TABLE_DIR = "pdf_tables"

@app.get("/assets/{filename}")
async def get_asset(filename: str):
    """Serves images/tables (assets) from the backend folder."""
    # Basic check to prevent directory traversal attacks
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    # Check both the image and table directories
    asset_path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(asset_path):
        asset_path = os.path.join(TABLE_DIR, filename)
        
    if os.path.exists(asset_path):
        # NOTE: You need to correctly determine the media type (content-type)
        from fastapi.responses import FileResponse
        return FileResponse(asset_path)
    else:
        raise HTTPException(status_code=404, detail="Asset not found.")