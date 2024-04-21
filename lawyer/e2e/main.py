from fastapi import FastAPI
from routes import prompt, parameter, chatbot, user
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot.router)
app.include_router(prompt.router)
app.include_router(parameter.router)
app.include_router(user.router)


@app.get("/")
def root():
    return {"message": "Hello World"}

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main())
    main()
