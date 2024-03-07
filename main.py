from Qa_inference import QA_inference

from langchain_community.vectorstores import FAISS
from typing import AsyncGenerator, NoReturn
import json
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse


from langchain_community.embeddings import HuggingFaceEmbeddings
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage



app = FastAPI()

embedding = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")



db = FAISS.load_local("faiss_NHS", embedding)

retriever= db.as_retriever(search_kwargs={'k': 6, 'lambda_mult': 0.5}, search_type="similarity")

api_key = "cIzixjEaY6GqWlugetGtVex3uN6vtmot"

client = MistralAsyncClient(api_key=api_key)

qa = QA_inference(model = client, retriver= retriever)
print("loaded")



@app.get("/")
async def web_app():
    """
    Web App
    """
    return {"status" : 1}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> NoReturn:
    """
    Websocket for AI responses
    """
    await websocket.accept()
    
        
    while True:
        message = await websocket.receive_text()

        template = f"""You are a translator assistant. Your only goal is to translate french text in english
        You are only allow to generate the english translation and nothing else.
        For example, if the text is: 'Quels sont les symptomes du covid?' you have to output 'What are the covid symptoms' and nothing else.
        Here is the text to translate: 
        {message}"""     
        print("translate :")
        messages = [
        ChatMessage(role="system", content=template),
        ChatMessage(role="user", content="translate the text.")
        ]

        async_response = client.chat_stream(
            model="mistral-medium",
            messages=messages)
        english_query = ''
        async for chunk in async_response:
            content = chunk.choices[0].delta.content
            if content:
                english_query += content

        print(english_query)
        documents, results = qa.get_documents(message)
        
        print("document type", type(documents))
        print("result type :", type(results))
        await send_text(websocket, results, content_type='document')
        
    # Send message
       # await send_text(websocket, str(results), content_type='document')
    
    # Send documents
        async for text in qa.completion(message, documents, results):
            await send_text(websocket, text, content_type='message')


async def send_text(websocket, content, content_type):
        # Your custom logic for handling different content types
        if content_type == 'message':
        # Handle message
            await websocket.send_text(json.dumps({"type": "message", "content": content}))

        elif content_type == 'document':

        # Handle document
            for doc in content:
                await websocket.send_text(json.dumps({"type": "documents", "content":doc.page_content, "uri" : doc.metadata["source_link"], "name" : doc.metadata["maladie"]} ))
       


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Use "0.0.0.0" to make it accessible externally
        port=8000,       # Change the port as needed
        log_level="debug",
        reload=True,
        reload_dirs=["./"],  # Optional: specify directories to watch for changes
        reload_delay=2,      # Optional: set a delay for the automatic reload
        reload_includes=["app.*"],
    )
