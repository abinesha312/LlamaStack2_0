import os
import torch
from torch import nn
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

DB_FAISS_PATH = 'vectorstore/db_faiss'

# custom_prompt_template = """
# You are an expert assistant for the University of North Texas (UNT). 
# Use the following pieces of information, scraped from official UNT sources, 
# to answer the user's question.

# Please ensure that all your answers relate specifically to the University of North Texas 
# and its departments. If the information is not available in the provided context, simply state, 
# "I'm unable to find that information from the available resources."

# Context: {context}
# Question: {question}

# Provide a concise and helpful answer based on the University of North Texas information:

# Helpful answer:
# """



custom_prompt_template = """
You are an expert assistant for the University of North Texas (UNT). 
Use the following pieces of information, scraped from official UNT sources, 
to answer the user's question.

Here is how you can do for better answers.
Think step-by-step. First, review the Question, analyzing for key points. Next, print your grocery list.
Then, search the Internet and Scrapped data from the UNT Officials for information about the Questiosn which are asked
and the working point of view available and map that information on to the grocery list.



Please ensure that all your answers relate specifically to the University of North Texas 
and its departments. If the information is not available in the provided context, simply state, 
"I'm unable to find that information from the available resources."

Context: {context}
Question: {question}

Provide a concise and helpful answer based on the University of North Texas information:

Finally, put this information into a table and cite your sources, including links in a separate column on the table
linking to an internet-based resources.


Helpful answer:


"""


def set_custom_prompt():

    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=['context', 'question']
    )
    return prompt

def load_llm():
    model_path = '/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'
 
    try:
        hf_pipeline = pipeline('text-generation', model=model_path, device=0)
        generator = HuggingFacePipeline(pipeline=hf_pipeline)
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def check_gpus():
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    print(f"Number of GPUs available: {num_gpus}")
    print(f"Current GPU being used: {current_device}")
    return num_gpus

generator = load_llm()

def chaatBot_Mode_chain(generator, prompt, db, memory):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=generator,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

def handle_user_message(message, chat_history):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda:0'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    qa = chaatBot_Mode_chain(generator, qa_prompt, db, memory)
    prnt = check_gpus()
    print("The total number of GPUS  -",prnt) 

    response = qa({'question': message})
    helpful_answer_marker = "Helpful answer:\n"
    answer_start = response['answer'].find(helpful_answer_marker)
    if answer_start != -1:
        helpful_answer = response['answer'][answer_start + len(helpful_answer_marker):].strip()
    else:
        helpful_answer = "Helpful answer not found."
    print("Helpful Answer:", helpful_answer)
    chat_history.append((message, helpful_answer))
    
    return "", chat_history

model = nn.DataParallel(generator)
model.to('cuda')