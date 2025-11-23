
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
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
    manufacturer: str
    model: str
    mode: str = "owner"
    component: Optional[str] = None  # NEW: Added to support component-focused queries


# NEW FUNCTION: Enhance query based on mode and component
def enhance_prompt_by_mode(query: str, mode: str, component: str = None) -> str:
    """
    Enhance the prompt based on the mode to get different response styles
    """
    
    if mode == "mechanic":
        system_prompt = """You are a professional automotive technician assistant. 
When answering:
1. Provide DETAILED technical specifications with exact measurements
2. Include step-by-step procedures numbered clearly
3. Reference specific page numbers from the manual when available
4. Include torque specifications, part numbers, and tool requirements
5. Add safety warnings and precautions
6. If diagrams or tables exist in the manual, explicitly mention and include them
7. Use technical terminology appropriate for professional mechanics
8. Include troubleshooting steps and diagnostic procedures

Format your response with clear sections:
- Technical Specifications
- Required Tools & Parts
- Step-by-Step Procedure
- Safety Precautions
- Common Issues & Solutions
"""
    else:  # owner mode
        system_prompt = """You are a friendly vehicle owner assistant.
When answering:
1. Use SIMPLE, easy-to-understand language
2. Avoid technical jargon unless necessary, then explain it
3. Focus on practical, actionable advice
4. Include visual references when available
5. Provide safety tips in plain language
6. Break down complex procedures into simple steps
7. Add helpful analogies to make concepts clear
8. Emphasize when to seek professional help

Format your response with clear sections:
- Quick Answer (1-2 sentences)
- Detailed Explanation (simple language)
- Visual Aids (if available)
- Safety Tips
- When to See a Mechanic
"""
    
    if component:
        component_focus = f"\n\nFOCUS AREA: The user is specifically asking about the {component} component. Prioritize information related to this component."
        system_prompt += component_focus
    
    # Combine system prompt with user query
    enhanced_query = f"{system_prompt}\n\nUser Question: {query}\n\nProvide a comprehensive answer from the vehicle manual, including any relevant diagrams, tables, or specifications."
    
    return enhanced_query


@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    """
    Handle user queries with manufacturer and model context
    """
    try:
        # MODIFIED: Enhance the query based on mode and component
        enhanced_query = enhance_prompt_by_mode(
            query=request.query,
            mode=request.mode,
            component=request.component
        )
        
        # Use enhanced query instead of original query
        result = query_pinecone_for_answer(
            query=enhanced_query,  # CHANGED: Using enhanced_query instead of request.query
            manufacturer=request.manufacturer,
            model_name=request.model,
            mode=request.mode
        )
        
        return {
            "status": "success",
            "answer": result
        }
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

ASSET_DIR = "pdf_assets"
TABLE_DIR = "pdf_tables"

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # # main.py
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import shutil
# import os

# # Import your core logic functions
# from basic import query_pinecone_for_answer, upload_pdf_to_pinecone

# app = FastAPI()

# origins = [
#     "http://localhost:3000",      
#     "http://localhost:5173",      
#     "http://127.0.0.1:5173",     
# ]


    

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # allow all origins temporarily for testing
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# class QueryRequest(BaseModel):
#     query: str
#     manufacturer: str  # NEW
#     model: str         # NEW
#     mode: str = "owner"



# # @app.post("/api/ask", response_model=Dict[str, Any])
# # async def ask_question(request: QueryRequest):
# #     """Endpoint for the chatbot query."""
# #     try:
# #         answer = query_pinecone_for_answer(request.query, request.mode)
# #         return {"answer": answer}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# # @app.post("/api/upload")
# # async def upload_manual(file: UploadFile = File(...)):
# #     """Endpoint to upload a PDF and process it into Pinecone."""
# #     if not file.filename.endswith('.pdf'):
# #         raise HTTPException(status_code=400, detail="Only PDF files are supported.")

# #     try:
# #         # Read file content
# #         content = await file.read()
        
# #         # Process and upload the file using the core logic
# #         result = upload_pdf_to_pinecone(content, file.filename)
        
# #         return {"message": "PDF processed and uploaded successfully.", "details": result}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Upload and processing failed: {e}")
# @app.post("/api/ask")
# async def ask_question(request: QueryRequest):
#     """
#     Handle user queries with manufacturer and model context
#     """
#     try:
#         result = query_pinecone_for_answer(
#             query=request.query,
#             manufacturer=request.manufacturer,
#             model_name=request.model,
#             mode=request.mode
#         )
        
#         return {
#             "status": "success",
#             "answer": result
#         }
    
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# ASSET_DIR = "pdf_assets"
# TABLE_DIR = "pdf_tables"
# @app.get("/api/health")
# async def health_check():
#     return {"status": "healthy"}




# @app.get("/assets/{filename}")
# async def get_asset(filename: str):
#     """Serves images/tables (assets) from the backend folder."""
#     # Basic check to prevent directory traversal attacks
#     if ".." in filename or "/" in filename:
#         raise HTTPException(status_code=400, detail="Invalid filename.")
        
#     # Check both the image and table directories
#     asset_path = os.path.join(ASSET_DIR, filename)
#     if not os.path.exists(asset_path):
#         asset_path = os.path.join(TABLE_DIR, filename)
        
#     if os.path.exists(asset_path):
#         # NOTE: You need to correctly determine the media type (content-type)
#         from fastapi.responses import FileResponse
#         return FileResponse(asset_path)
#     else:
#         raise HTTPException(status_code=404, detail="Asset not found.")
    
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# main.py