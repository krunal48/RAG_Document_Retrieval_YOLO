from typing import Optional
from dataclasses import dataclass
from openai import OpenAI
from app.settings import SETTINGS
import json
import os
import base64
import gradio as gr
from agents import Agent, ModelSettings, function_tool,Runner, RunContextWrapper
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor





CLIENT = OpenAI()

@dataclass
class Query():
    query: str
    index: object
    client: object
    k: int
    context: list[str]
    links: dict

@dataclass
class Context():
    citations: list[str]

@dataclass
class AgentRequest:
    text: str
    patient_id: Optional[str] = None

@dataclass
class AgentResponse:
    reply: str
    citations: str
    agent: str
    images: dict
    def get_response(self):
        return self.reply + self.citations

def get_embeddings(text, model):
    return CLIENT.embeddings.create(input=text, model=model).data[0].embedding

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@function_tool
def get_context_paragraph(wrapper: RunContextWrapper[Query], k:int):
    embed_model = 'text-embedding-3-small'
    print(f'Model decided to gather {k} additional paragraphs')
    k += wrapper.context.k
    query_embeddings = get_embeddings(wrapper.context.query, model=embed_model)
    pinecone_response = wrapper.context.index.query(vector=query_embeddings,
                                    top_k=k,
                                    include_metadata=True,
                                    namespace='paragraph')
    contexts = [item['metadata']['text'] for item in pinecone_response['matches']]
    contexts = contexts[wrapper.context.k:]

    wrapper.context.k = k
    wrapper.context.context += contexts

    return contexts

@function_tool
async def get_context_images(wrapper: RunContextWrapper[Query]):
    embed_model = 'text-embedding-3-small'
    k = 3
    print(f'Model decided to gather images')
    query_embeddings = get_embeddings(wrapper.context.query, model=embed_model)
    pinecone_response = wrapper.context.index.query(vector=query_embeddings,
                                    top_k=k,
                                    include_metadata=True,
                                    namespace='captioned image')

    contexts = [item['metadata']['text'] for item in pinecone_response['matches']]
    images = [item['metadata']['image'] for item in pinecone_response['matches']]

    image_contexts, links = await ask_gpt(wrapper,wrapper.context.query,images,contexts, len(wrapper.context.context))

    return image_contexts

def caption_image(wrapper: RunContextWrapper[Query], query, image_url_, caption, contexts, links, k , multi_modal_model="gpt-5-nano"):
    print(f'Captioning: {image_url_}')
    base64_image = encode_image_to_base64(image_url_)
    response = CLIENT.responses.create(
        model=multi_modal_model,
        input=[{
            "role":"user",
            "content":[
                {"type":"input_text",
                "text": f"Caption:\n{caption}\nQuery:\n{query}\nDescribe the contents of this image in regards to its caption and query. Do not answer the query, answer in one paragraph"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                },
            ],
        }]
    )
    wrapper.context.k += 1
    wrapper.context.links[wrapper.context.k] = image_url_
    wrapper.context.context.append(response.output_text)
    print(f'Captioned: {image_url_}')
    time.sleep(1)

async def ask_gpt(wrapper: RunContextWrapper[Query], query,images,captions, k):
    contexts = []
    links = {}
    print('here')
    # for i, im in enumerate(images):
    #     im = im.split('/')
    #     im = os.path.join(*im)
    #     contexts.append(caption_image(query,im,captions[i]))
    #     links[k+i+1] = im
    # threads = []
    # for i in range(0,k):
    #     threads.append(threading.Thread(
    #                                     target=caption_image,
    #                                     args=(query,
    #                                         images[i],
    #                                         captions[i],
    #                                         contexts,
    #                                         links,
    #                                         k
    #                                         )
    #                                     )
    #                 )
    #     threads[i].start()
    #     #time.sleep(1)
    # for thread in threads:
    #     thread.join()

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        tasks = []
        for i in range(0,k):
            loop.run_in_executor(executor, 
                                caption_image,
                                wrapper,
                                query,
                                images[i],
                                captions[i],
                                contexts,
                                links,
                                k
                                )
    print('task start')
    await asyncio.gather(*tasks)
    print(f'Image Contexts: \n {contexts}')
    print(f'Image Links: \n {links}')
    return contexts, links

class AgenticRAG:
    name = "AgenticRAG"
    def __init__(self, index, k=5):
        self.index = index
        self.client = OpenAI()
        self.k = k
        self.agent = Agent[Query](
                            name='RagAgent',
                            instructions='Your job is to help answer questions by providing more citations as necessary based on the query and current citations.',
                            model='gpt-4.1-mini',
                            tools=[get_context_paragraph, get_context_images],
                            output_type=Context,
                            model_settings=ModelSettings(tool_choice="auto")
                          )
        self.links = dict()



    def get_context(self, query, embed_model='text-embedding-3-small'):
        query_embeddings = get_embeddings(query, model=embed_model)
        pinecone_response = self.index.query(vector=query_embeddings,
                                        top_k=self.k,
                                        include_metadata=True,
                                        namespace='paragraph')
        
        contexts = [item['metadata']['text'] for item in pinecone_response['matches']]
        return contexts

    #An Agentic RAG manager solution that gathers context if it deems it necessary
        #Starts with DEFK entries from paragraph namespace
        #Decides if it wants to add more from paragraphs into the context
        #Decides if it wants to add captioned images to the context
    async def gather_context(self, query) -> list:
        context = self.get_context(query)
        # messages = [{"role": "user",
        #               "content": f"the query: {query} \n the Current Context: {context[0]}\n\nInstructions: \nIf the context above does not provide enough information to answer the above query, use the tools provided to gather additional context to answer.\n"}]
        links = {}
        #print(f'Initial Contexts: {context}')
        agent_context = Query(query=query, 
                              index=self.index, 
                              client=self.client, 
                              k=5, 
                              context=context, 
                              links=links)
        response = await Runner.run(starting_agent=self.agent,
                                    context=agent_context,
                                    input=f"the query: {query} \n the Current Citations: {agent_context.context}\n\nInstructions: \nIf the current citations above does not provide enough information to answer the above query, use the tools provided to gather additional citations to answer.\n"
                                    )
        #print(f'Context Object:\n {agent_context}')
        print(f'Final output:\n{response.final_output}')
        #print(f'New Items:\n{response.new_items}')
        #print(f'Raw responses:\n{response.raw_responses}')
        #print(f'Context Object:\n {response.context_wrapper}')
        #print(f'Context Context Context:\n{response.context_wrapper.context.context}')
        # contexts += response.citations
        self.links.update(agent_context.links)
        return agent_context.context

    def answer(self, query: str) -> AgentResponse:
        context = asyncio.run(self.gather_context(query))
        citations = '\n\n\n-----Citations-----\n\n'
        images = []
        for i, c in enumerate(context):
            citations += '\n\n['+str(i+1)+'] '+c
        print(citations)
        for i in self.links.keys():
            images.append([self.links[i],'['+str(i)+']'])
        print(images)
        if not context:
            return AgentResponse("No documents in the vector store yet.", [], self.name)
        prompt = f"{context}\n\nQuestion: {query}\nAnswer the question using the' \
                 ' the citations and place their corresponding' \
                 ' citation number in brackets like [1] [2] [3] etc... after each statement where that citation was used.\n" \
                 ' Dont directly quote the citations. Summarize their key points in your own terms.'
        resp = self.client.chat.completions.create(
            model='gpt-4.1-mini', temperature=0.8,
            messages=[{"role":"system","content":SETTINGS.system_prompt},{"role":"user","content":prompt}]
        )
        ans = resp.choices[0].message.content.strip()
        return AgentResponse(ans, citations, self.name, images)
