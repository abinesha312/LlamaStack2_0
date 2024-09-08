import os
import torch
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
import mesop as me
import mesop.labs as mel
from AuthRedis.auth import Auth
from AuthRedis.login import login_page
from AuthRedis.register import register_page

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DB_FAISS_PATH = 'vectorstore/db_faiss'

CUSTOM_PROMPT_TEMPLATE = """
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
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['context', 'question'])
    return prompt

def load_llm():
    model_path = '/home/haridoss/Models/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'
    try:
        hf_pipeline = pipeline('text-generation', model=model_path, device=0)
        generator = HuggingFacePipeline(pipeline=hf_pipeline)
        print("Model loaded successfully", generator)
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def check_gpus():
    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    print(f"Number of GPUs available: {gpu_count}")
    print(f"Current GPU being used: {current_gpu}")
    return gpu_count

def create_chatbot_chain(llm, prompt, db, memory):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    @classmethod
    def from_dict(cls, data):
        return cls(role=data['role'], content=data['content'])

    def to_dict(self):
        return {"role": self.role, "content": self.content}

def handle_user_message(message, chat_history):
    if chat_history is None:
        chat_history = []

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 1})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    llm = load_llm()
    qa = create_chatbot_chain(llm, qa_prompt, db, memory)
    gpu_count = check_gpus()
    print("Total number of GPUs:", gpu_count)

    response = qa({'question': message})
    answer_marker = "Helpful answer:\n"
    answer_start = response['answer'].find(answer_marker)
    if answer_start != -1:
        helpful_answer = response['answer'][answer_start + len(answer_marker):].strip()
    else:
        helpful_answer = "Helpful answer not found."
    print("Helpful Answer:", helpful_answer)

    chat_history.append(ChatMessage(role="user", content=message))
    chat_history.append(ChatMessage(role="assistant", content=helpful_answer))
    
    return "", chat_history

def is_authenticated():
    return hasattr(me.state, 'session_id') and bool(me.state.session_id)

@me.page(path="/", title="UNT Chatbot")
def main_page():
    me.text(f"Is authenticated: {is_authenticated()}")  # Debug message
    me.text(f"Current session ID: {getattr(me.state, 'session_id', 'None')}")  # Debug message
    if not is_authenticated():
        me.text("navigateing to login")  # Debug message
        me.navigate("/login")
    else:
        me.text("Loading chatbot page")  # Debug message
        chatbot_page()

@me.page(path="/logout", title="Logout")
def logout_page():
    if hasattr(me.state, 'session_id'):
        delattr(me.state, 'session_id')
    me.navigate("/login")

@me.page(path="/chatbot", title="UNT Chatbot")
def chatbot_page():
    if not is_authenticated():
        me.navigate("/login")
    else:
        mel.chat(transform, title="UNT Chatbot", bot_user="Chatbot")

def transform(user_input: str, history: list) -> str:
    if not history:
        history = []
    chat_history = [ChatMessage(**msg) if isinstance(msg, dict) else msg for msg in history]
    _, updated_history = handle_user_message(user_input, chat_history)
    return updated_history[-1].content if updated_history else ""

def main():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    
    global generator
    generator = load_llm()
    print("Step 1: Generators initiated.", generator)
    me.run()

if __name__ == "__main__":
    main()