import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# DDP Initialization
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
You are an expert assistant for the University of North Texas (UNT).
Use the following pieces of information, scraped from official UNT sources, 
to answer the user's question.

Here's how you can improve your answers:
1. Think step-by-step. First, review the question, analyzing the key points.
2. Next, create a grocery list of the essential information needed.
3. Then, search the vector database and scraped data from UNT officials for relevant information related to the questions asked.
4. Align the gathered information with your grocery list, ensuring it covers all key points. 
5. Finally, include the source URL for every piece of information you find.

Please ensure that all your answers relate specifically to the University of North Texas 
and its departments. If the information is not available in the provided context, simply state, 
"I'm unable to find that information from the available resources."

Context: {context}
Question: {question}

Provide a concise and helpful answer based on the University of North Texas information:

Finally, present this information in a table format, and include a separate column in the table linking to the internet-based resources where the information was found.

Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=['context', 'question']
    )
    return prompt

def load_llm(rank):
    model_path = '/home/haridoss/Models/Models/llama-models/models/llama3_1/Meta-Llama-3.1-70B-Instruct'
    try:
        hf_pipeline = pipeline('text-generation', model=model_path, device=rank)
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

def chaatBot_Mode_chain(generator, prompt, db, memory):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=generator,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

def handle_user_message(message, chat_history, rank):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': f'cuda:{rank}'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    qa = chaatBot_Mode_chain(generator, qa_prompt, db, memory)
    prnt = check_gpus()
    print("The total number of GPUS  -", prnt) 

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

def main(rank, world_size):
    setup_ddp(rank, world_size)
    
    # Configure memory allocation for each process
    torch.cuda.set_per_process_memory_fraction(0.5, device=rank)
    
    generator = load_llm(rank)
    
    # Wrap the model with DDP
    ddp_model = DDP(generator, device_ids=[rank])
    
    # Call the chatbot chain
    chat_history = []
    # Example user input
    user_message = "What are the research opportunities at UNT?"
    response, chat_history = handle_user_message(user_message, chat_history, rank)

    cleanup_ddp()

if __name__ == "__main__":
    world_size = 8  # Adjust based on the number of GPUs you have
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)




# import os
# import torch
# from torch import nn
# from transformers import pipeline
# from langchain.prompts import PromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.llms import HuggingFacePipeline
# #--- DDP


# DB_FAISS_PATH = 'vectorstore/db_faiss'

# custom_prompt_template = """
# You are an expert assistant for the University of North Texas (UNT).
# Use the following pieces of information, scraped from official UNT sources, 
# to answer the user's question.

# Here's how you can improve your answers:
# 1. Think step-by-step. First, review the question, analyzing the key points.
# 2. Next, create a grocery list of the essential information needed.
# 3. Then, search the vector database and scraped data from UNT officials for relevant information related to the questions asked.
# 4. Align the gathered information with your grocery list, ensuring it covers all key points. 
# 5. Finally, include the source URL for every piece of information you find.

# Please ensure that all your answers relate specifically to the University of North Texas 
# and its departments. If the information is not available in the provided context, simply state, 
# "I'm unable to find that information from the available resources."

# Context: {context}
# Question: {question}

# Provide a concise and helpful answer based on the University of North Texas information:

# Finally, present this information in a table format, and include a separate column in the table linking to the internet-based resources where the information was found.

# Helpful answer:
# """
 


# def set_custom_prompt():

#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=['context', 'question']
#     )
#     return prompt

# def load_llm():
#     model_path = '/home/haridoss/Models/Models/llama-models/models/llama3_1/Meta-Llama-3.1-70B-Instruct' #'/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'
 
#     try:
#         hf_pipeline = pipeline('text-generation', model=model_path, device=0)
#         generator = HuggingFacePipeline(pipeline=hf_pipeline)
#         return generator
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise

# def check_gpus():
#     num_gpus = torch.cuda.device_count()
#     current_device = torch.cuda.current_device()
#     print(f"Number of GPUs available: {num_gpus}")
#     print(f"Current GPU being used: {current_device}")
#     return num_gpus

# generator = load_llm()

# def chaatBot_Mode_chain(generator, prompt, db, memory):
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=generator,
#         retriever=db.as_retriever(search_kwargs={'k': 2}),
#         return_source_documents=True,
#         memory=memory,
#         combine_docs_chain_kwargs={'prompt': prompt}
#     )
#     return qa_chain

# def handle_user_message(message, chat_history):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda:0'})

#     db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

#     qa_prompt = set_custom_prompt()
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

#     qa = chaatBot_Mode_chain(generator, qa_prompt, db, memory)
#     prnt = check_gpus()
#     print("The total number of GPUS  -",prnt) 

#     response = qa({'question': message})
#     helpful_answer_marker = "Helpful answer:\n"
#     answer_start = response['answer'].find(helpful_answer_marker)
#     if answer_start != -1:
#         helpful_answer = response['answer'][answer_start + len(helpful_answer_marker):].strip()
#     else:
#         helpful_answer = "Helpful answer not found."
#     print("Helpful Answer:", helpful_answer)
#     chat_history.append((message, helpful_answer))
    
#     return "", chat_history

# model = nn.DataParallel(generator)
# model.to('cuda')