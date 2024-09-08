import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
import mesop as me
import mesop.labs as mel
from mesop import stateclass
from dataclasses import dataclass, field
from datetime import datetime

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

def load_llm():
    model_path = '/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def check_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs available.")
    return num_gpus

model, tokenizer = load_llm()

def chaatBot_Mode_chain(model, tokenizer, prompt, db, memory):
    generator = HuggingFacePipeline(model=model, tokenizer=tokenizer)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=generator,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

@dataclass
class Session:
    user_id: str = ''
    created_at: str = ''
    chat_history: list = field(default_factory=list)

def initialize_session(user_id):
    return Session(user_id=user_id, created_at=datetime.now().isoformat())

@stateclass
class State:
    session: Session = field(default_factory=lambda: initialize_session(user_id="default_user"))

def update_chat_history(state, input_msg, response_msg):
    state.session.chat_history.append((input_msg, response_msg))

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/",
    title="Mesop Demo Chat",
)
async def page():
    state = me.state(State)  # Access the session state
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda:0'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa_chain = chaatBot_Mode_chain(model, tokenizer, qa_prompt, db, memory)

    # Initialize chat and set session state
    state.session = initialize_session(user_id="user_123")  # Initialize session with a specific user ID
    await mel.chat(lambda input, history: transform(input, history, qa_chain, state), 
                   title="LangChain Chat using Meta Llama3 for Model Inference", 
                   bot_user="Meta Llama3 Mesop Bot")

async def transform(input: str, history: list[mel.ChatMessage], qa_chain, state):
    # Process the input and generate a response
    response = qa_chain({'question': input})
    helpful_answer_marker = "Helpful answer:\n"
    answer_start = response['answer'].find(helpful_answer_marker)
    if answer_start != -1:
        helpful_answer = response['answer'][answer_start + len(helpful_answer_marker):].strip()
    else:
        helpful_answer = "Helpful answer not found."
    
    # Update chat history
    update_chat_history(state, input, helpful_answer)
    return helpful_answer

if __name__ == "__main__":
    check_gpus() 
    me.run()


# from langchain import PromptTemplate
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import LLMChain, create_retrieval_chain
# import chainlit as cl
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# from mesop import stateclass

# import mesop as me
# import mesop.labs as mel

# DB_FAISS_PATH = 'vectorstore/db_faiss'

# custom_prompt_template = """You are an expert assistant for the University of North Texas (UNT).
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
#     """
#     The Prompt template for Chat Bot Mode for each vectorstore
#     """
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=['context', 'question']
#     )
#     return prompt

# def chaatBot_Mode_chain(model, tokenizer, prompt, db, memory):
#     retriever = db.as_retriever(search_kwargs={'k': 2})
#     question_generator_chain = LLMChain(llm=model, prompt=prompt)
#     qa_chain = create_retrieval_chain(
#         retriever=retriever,
#         question_generator=question_generator_chain,
#         memory=memory,
#         return_source_documents=True
#     )
#     return qa_chain

# def load_llm():
#     model_path = '/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     return model, tokenizer

# def create_qa_chain():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
#     # Load the model and tokenizer
#     model, tokenizer = load_llm()
#     qa_prompt = set_custom_prompt()
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=model,
#         retriever=db.as_retriever(search_kwargs={'k': 2}),
#         return_source_documents=True,
#         memory=memory,
#         combine_docs_chain_kwargs={'prompt': qa_prompt},
#         tokenizer=tokenizer
#     )
#     return qa_chain

# def check_gpus():
#     num_gpus = torch.cuda.device_count()
#     if num_gpus > 0:
#         for i in range(num_gpus):
#             gpu_name = torch.cuda.get_device_name(i)
#             print(f"GPU {i}: {gpu_name}")
#     else:
#         print("No GPUs available.")
#     return num_gpus

# def final_result(query):
#     qa_result = create_qa_chain()
#     response = qa_result({'query': query})
#     return response


# @dataclass
# class Session:
#     user_id: str = ''
#     created_at: str = ''
#     chat_history: list = field(default_factory=list)

# def initialize_session(user_id):
#     return Session(user_id=user_id, created_at=datetime.now().isoformat())



# @stateclass
# class State:
#     session: Session = field(default_factory=lambda: initialize_session(user_id="default_user"))

# def update_chat_history(state, input_msg, response_msg):
#     state.session.chat_history.append((input_msg, response_msg))

# @me.page(
#     security_policy=me.SecurityPolicy(
#         allowed_iframe_parents=["https://google.github.io"]
#     ),
#     path="/",
#     title="Mesop Demo Chat",
# )

# @cl.on_chat_start
# async def start():
#     chain = create_qa_chain()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to the Bot. What is your query?"
#     await msg.update()
#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message):
#     chain = cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["answer"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\nSources:" + str(sources)
#     else:
#         answer += "\nNo sources found"

#     await cl.Message(content=answer).send()

# if __name__ == "__main__":
#     check_gpus() 
#     cl.run()

# ###############################################################################################






# @me.page(
#     security_policy=me.SecurityPolicy(
#         allowed_iframe_parents=["https://google.github.io"]
#     ),
#     path="/",
#     title="Mesop Demo Chat",
# )
# def page():
#     tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", clean_up_tokenization_spaces=True)
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda:0'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
#     qa_prompt = set_custom_prompt()
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
#     qa_chain = chaatBot_Mode_chain(generator, qa_prompt, db, memory)

#     mel.chat(lambda input, history: transform(input, history, qa_chain, tokenizer), 
#              title="LangChain Chat using Meta Llama3 for Model Inference", 
#              bot_user="Meta Llama3 Mesop Bot")

# def transform(input: str, history: list[mel.ChatMessage], qa_chain, tokenizer):
#     global generator
#     # Tokenize the input
#     tokenized_input = tokenizer(input, return_tensors="pt", clean_up_tokenization_spaces=True).to('cuda')
    
#     # Process the input and generate a response
#     response = qa_chain({'question': input})
#     helpful_answer_marker = "Helpful answer:\n"
#     answer_start = response['answer'].find(helpful_answer_marker)
#     if answer_start != -1:
#         helpful_answer = response['answer'][answer_start + len(helpful_answer_marker):].strip()
#     else:
#         helpful_answer = "Helpful answer not found."
    
#     return helpful_answer

# if __name__ == "__main__":
    
#     me.run()