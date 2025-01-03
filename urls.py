from pydantic import BaseModel
from fastapi import APIRouter

from retriever import get_response, get_competitor

router = APIRouter()

class QueryInp(BaseModel):
    query: str

@router.post("/chat")
async def chat(inp: QueryInp):
    response = await get_response(inp.query)
    
    return {
        "response": response
    }
    
@router.post("/competitor")
async def competitor(inp: QueryInp):
    response = await get_competitor(inp.query)
    
    return {
        "response": response
    }