
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from mistralai.models.chat_completion import ChatMessage



class QA_inference:

  def __init__(self, model, retriver):
    self.client = model
    self.retriever = retriver

  def get_documents(self,query):
    results = self.retriever.get_relevant_documents(query)
    documents = ''
    for i in range(len(results)):
        documents += f"Document {i} \n " + str(results[i].page_content) + "\n"
    
    return documents, results

  async def completion(self, query, documents, results):
   
    print(results)
    template = f""""Vous êtes un assistant qui doit répondre aux questions des utilisateurs en se basant sur les documents.
    Vous ne devez vous fier qu'aux documents et rien d'autre.
    Si la réponse ne se trouve pas dans les documents, indiquez que vous ne pouvez pas répondre, et c'est tout.
    Pour les termes techniques, fournissez une explication comme si l'utilisateur avait 5 ans.
    Vous devez répondre seulement en Français.
    Voici les documents :
    {documents}
    """

        
    print("Iris :")
    messages = [
    ChatMessage(role="system", content=template),
    ChatMessage(role="user", content=query)
    ]

    async_response = self.client.chat_stream(model="mistral-medium", 
                                            temperature=0.1,
                                            max_tokens=1024,
                                            top_p=1,
                                            
                                            messages=messages)
    all_content =""
    async for chunk in async_response:
        content = chunk.choices[0].delta.content
        if content:
            all_content += content
            yield all_content
 
      