# app/ui/app.py
# --- path bootstrap so running as a script works ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------

###Imports###
import os, io, json, tempfile
import gradio as gr
from openai import OpenAI

from app.agents.ExtractAgent import ExtractAgent
from app.settings import SETTINGS
from app.vector_store.pinecone_adapter import PineconeVectorStore
from app.agents.AgenticRag import AgenticRAG, AgentRequest
from app.agents.asha import ASHAAgent
from app.agents.embryology import EmbryologyStore, EmbryologyResultAgent, now_utc
from app.yolo.yolo import YoloExtractor
from pinecone import Pinecone
import numpy as np
import pandas as pd
from agents.exceptions import InputGuardrailTripwireTriggered
import cv2

class file_queue:
    file_names: list[str]
    def clear_queue(self):
        self.file_names = []

###Constants/ globals###
PATH = os.path.dirname(os.path.abspath(__file__))+'\\'+'cache\\'
FILE_PATH = PATH 
IMAGES_PATH = PATH + '/images/'
TEMP_PATH = PATH + '/temp/'
CURRENT = ''
fq = file_queue()
fq.clear_queue()
_pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
_pine_name = os.environ.get('PINECONE_NAME')
_index = _pc.Index(_pine_name)
rag = AgenticRAG(_index)
ext = ExtractAgent()

###helpers###
#Read the inputted file and transform it into a byte representation.
def _read_bytes_from_any(file_like) -> bytes:
    if isinstance(file_like, bytes):
        return file_like
    if isinstance(file_like, str):
        with open(file_like, "rb") as fh:
            return fh.read()
    if hasattr(file_like, "read"):
        return file_like.read()
    raise TypeError(f"Unsupported file type: {type(file_like)}")

#Delete all files in the documents folder (Where input files are stored)
def clear_documents():
    print('Cleared Documents Folder!')
    path = FILE_PATH  + 'documents/'
    for i in os.listdir(path):
        if(not os.path.isdir(path+i)):
            os.remove(path+i)
    pass

###Rag###
#Call the agentic RAG, Guardrail in place to keep questions on topic.
def call_rag(message, history):
    try:
        ext.answer(message)
        agent_res = rag.answer(message)
    except InputGuardrailTripwireTriggered as e:
        return ['Please ask something relevant to medicine, fertility, or embryology'], []
    print(f'History: \n{history}')
    if(len(agent_res.images) != 0):
        print('Images Found!')
        print(agent_res.images)
    return agent_res.get_response(), agent_res.images

###Upload Data###
#Helps prevent code injection in the pinecone database
def prepare_DF(df):
  import json,ast
  try: df=df.drop('Unnamed: 0',axis=1)
  except: print('Unnamed Not Found')
  df['values']=df['values'].apply(lambda x: np.array([float(i) for i in x.replace("[",'').replace("]",'').split(',')]))
  df['metadata']=df['metadata'].apply(lambda x: ast.literal_eval(x))
  return df

#Converts a pandas dataframe to be a simple list of tuples, 
# formatted how the `upsert()` method in the Pinecone Python client expects.
def convert_data(chunk):
    data = []
    for i in chunk.to_dict('records'):
        data.append(i)
    return data

#Yields a series of slices of the original iterable, up to the limit of what size is.
def load_chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq.iloc[pos:pos + size]

#Uploads data into the vector database ysing pinecone (index)
def upsert(dfs, names, index):
    for i,df in enumerate(dfs):
        for load_chunk in load_chunker(df,200):
            index.upsert(vectors=convert_data(load_chunk),
                         namespace=names[i])

#Transfers files from the temp image store in /cache/temp/
# to permanent storage in /cache/images/
def save_images():
    for folder in os.listdir(TEMP_PATH):
        for image in os.listdir(os.path.join(TEMP_PATH,folder)):
            cv2.imwrite(filename=os.path.join(IMAGES_PATH,folder,image),
                        img=cv2.imread(os.path.join(TEMP_PATH,folder,image)))
        clear_images(TEMP_PATH+folder)
    pass

#Deletes images originating from a rejected file from the temp image store.
# This prevents temp images from becoming permanent.
def clear_rejected(file):
    pdf = file.split('.')[0]
    for folder in os.listdir(TEMP_PATH):
        for image in os.listdir(os.path.join(TEMP_PATH,folder)):
            if(not os.path.isdir(os.path.join(TEMP_PATH,folder)+'/'+image) and pdf in image):
                os.remove(os.path.join(TEMP_PATH,folder)+'/'+image)
    print(f"{file} Images Cleared!")

#Clears the cache/temp/ folder of images
def clear_images(folderpath):
    print(f'Cleared Temp Images Folder! - {folderpath}')
    for i in os.listdir(folderpath):
        if(not os.path.isdir(folderpath+'/'+i)):
            os.remove(folderpath+'/'+i)

#Handles uploading data retreived from YOLO to the pinecone vector database.
# ExtractAgent will reject any file irrelevant to medicine, fertility or embryology.
# Rejected files will be saved in a csv and not accepted in the future to save resources.
def upload_pinecone():
    print(f'Files in queue:\n {fq.file_names}')
    s_output = ''
    successful = []
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    pine_name = os.environ.get('PINECONE_NAME')
    index = pc.Index(pine_name)
    index_para = prepare_DF(pd.read_csv(PATH+'paragraph.csv'))
    index_capt = prepare_DF(pd.read_csv(PATH+'captioned image.csv'))

    if(os.path.exists(PATH+'file history.csv')):
        df_file_history = pd.read_csv(PATH+'file history.csv', index_col=False)
        print(f'File History Loaded:\n{df_file_history}')
    else:
        df_file_history = pd.DataFrame(columns=['File Name','Accepted', 'Reason'])

    for file in fq.file_names:
        if(len(df_file_history) == 0 or not file in df_file_history.index):
            global CURRENT 
            CURRENT = file
            print(f'Current File: {file}')
            series_df = index_para['metadata'].apply(pd.Series)
            series_df = series_df['text'].loc[series_df['image'].str.contains(file.split('.')[0])]
            try:
                print(f'Ext Output: \n{ext.answer(series_df.sum())}')
                df_file_history.loc[len(df_file_history)] = {'File Name':file,'Accepted':True,'Reason':'Relevant'}
                successful.append(file)
            except InputGuardrailTripwireTriggered as e:
                s_output += f"Irrelevant File: \n      {CURRENT} \nRelevancy Report:\n{e.guardrail_result.output.output_info}"
                df_file_history.loc[len(df_file_history)] = {'File Name':file,'Accepted':False,'Reason':'Irrelevant'}
                series_df = index_para['metadata'].apply(pd.Series)
                index_para = index_para.loc[~series_df['image'].str.contains(file.split('.')[0])]
                print(f'Updated Index Para: {index_para}')
                series_df = index_capt['metadata'].apply(pd.Series)
                index_capt = index_capt.loc[~series_df['image'].str.contains(file.split('.')[0])]
                print(f'Updated Index Capt: {index_capt}')
                clear_rejected(file)
        else:
            #Should not be possible to reach!
            print('Duplicate file detected!')
            assert False, 'Duplicate File Made it TOO far!'
    df_file_history.to_csv(PATH+'file history.csv', index=False)

    #series_desc = ext.analyze(index_para['metadata'].apply(pd.Series)['text'])
    #index_para = index_para.loc[series_desc]

    save_images()

    upsert([index_para, index_capt], 
           ['paragraph','captioned image'], 
           index)
    print('Indexes Uploaded to: ' + pine_name)

    if(len(successful) > 0):
        s_output = f'Accepted Documents:\n{successful}\n'

    return s_output

###Process Saved Documents###
#When the user presses the read documents button.
# Runs the YoloExtractor on the documents saved in the documents folder.
def read_documents(files):
    if(not files is None):
        output = save_files(files)
        return ['Unsaved files needed saving!:\n'+output[0]+'\n Processing now available.',gr.update(value=None)]
    s_output = ''
    try:
        yoloE = YoloExtractor(PATH, ext)
        indexes = yoloE.extract()
        print(indexes)
        s_output += upload_pinecone()
        s_output = "Documents Processed!\n" + s_output
        print(f'Read Documents Output:\n**{s_output}**')
        fq.clear_queue()
        return [s_output,gr.update(value=[])]
    except Exception  as e:
        print(e)
        return [f"Documents Processed Failed! \n {e} \n{fq.file_names} \n",gr.update(value=None)]
    finally:
        fq.clear_queue()
        clear_images(TEMP_PATH+'paragraph')
        clear_images(TEMP_PATH+'captioned image')
        print('Temp Files Deleted Successfully ')

###File Input###
#When the user presses the save button this function runs.
# Will save the files in the Gradio Files UI element.
# Will check these files against the file_history.csv to make sure these files are new.
def save_files(files):
    rejected = []
    if(os.path.exists(PATH+'file history.csv')):
        df_file_history = pd.read_csv(PATH+'file history.csv', index_col=False)
        print(f'File History Loaded:\n{df_file_history}')
    else:
        print('Creating File History File.')
        df_file_history = pd.DataFrame(columns=['File Name','Accepted', 'Reason'])
    try:
        for file in files:
            name = os.path.basename(file)
            if(len(df_file_history) == 0 or not name in df_file_history['File Name'].values and not name in fq.file_names):
                fq.file_names.append(name)
                raw = _read_bytes_from_any(file)
                print(FILE_PATH+name)
                with open(FILE_PATH+'documents\\'+name, "wb") as f:
                    f.write(raw)
            else:
                rejected.append(os.path.basename(file))
        df_file_history.to_csv(PATH+'file history.csv', index=False)
        s_output = ''
        if(len(fq.file_names) > 0):
            s_output += f" Saved Files:\n       {fq.file_names}\n Saved to documents folder.\n Ready to be processed.\n"
        if(len(rejected) > 0):
            s_output += f" Rejected Files:\n       {rejected}\n Are duplicates and were not saved.\n Please do not reupload duplicates."
        print(f'Save Documents UI Output:\n**{s_output}**')
        return [s_output,
                gr.update(value=None)]
    except Exception as e:
        clear_documents()
        return [f"PDF save error! \n {e}",gr.update(value=None)]

###Gradio UI###
#Uses gradio to build the user interface.
def build_ui():
    with gr.Blocks(title="Asha Fertility Assistant -Provider Portal-") as demo:
        gr.Markdown("## Asha Fertility Clinic  -Provider Portal-")
        #The tab for RAG chatbot
        with gr.Tab("Medisense Chat"):
            with gr.Row():
                with gr.Column():
                    #Gallery for images
                    gallery = gr.Gallery()
                with gr.Column():
                    #The chatbot interface
                    chatbot = gr.ChatInterface(
                                                fn=call_rag,
                                                chatbot=gr.Chatbot(height=500),
                                                additional_outputs=[gallery] #Image outputs returned to the gallery
                                                )
        #File upload tab
        with gr.Tab("Upload Documentation"):
            
            files = gr.File(label="Upload PDF/Image", file_count="multiple", type="filepath" )
            with gr.Row():
                save_button = gr.Button('Save Documents To Local Device')
                read_button = gr.Button('Process Saved Documents & Upload to Cloud')
            with gr.Row():
                out_msg = gr.Textbox(label='Output', lines=10, interactive=False)

            save_button.click(
                save_files,
                inputs=files,
                outputs=[out_msg,files]
            )
            read_button.click(
                read_documents,
                inputs=files,
                outputs=[out_msg,files]
            )
    return demo

###Main###
if __name__ == "__main__":

    #Create the folders for the input files.
    if(not os.path.exists(PATH)):
        os.mkdir(PATH)
    if(not os.path.exists(FILE_PATH)):
        os.mkdir(FILE_PATH)
    
    #Makes sure no old documents are in the folder.
    clear_documents()
    #Makes sure the file queue is empty and the queue attribute instantiated.
    fq.clear_queue()
    #Buildes the gradio UI
    demo = build_ui()
    #Get Gradio settings
    share = (os.getenv("GRADIO_SHARE", "false").lower() == "true")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    # preferred port from env; 0 means "auto-pick any free port"
    preferred_port = int(os.getenv("PORT", "7860"))

    try:
        demo.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
            server_port=preferred_port,
            share=share,
            show_error=True,
        )
    except OSError as e:
        # If the chosen port is busy, retry with an ephemeral free port
        if "Cannot find empty port" in str(e) or "Address already in use" in str(e):
            print(f"[INFO] Port {preferred_port} busy. Falling back to auto port...")
            demo.launch(
                server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=0,  # auto-pick
                share=share,
                show_error=True,
            )
        else:
            raise

"""
Enviroment Variables Used:

The gradio host
$env:NO_PROXY= "127.0.0.1,localhost"

gradio ui setting, set to not be available to other pcs.
$env:GRADIO_SHARE= "false"

name of the pinecone database
$env:PINECONE_NAME=

openai API key
$env:OPENAI_API_KEY=

pinecone API key
$env:PINECONE_API_KEY=
"""