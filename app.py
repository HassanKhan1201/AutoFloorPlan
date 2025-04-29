from fastapi import FastAPI, HTTPException
from fastapi_app.input_router import router as input_router

app = FastAPI()
app.include_router(input_router, prefix="/api", tags=["input"])

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)