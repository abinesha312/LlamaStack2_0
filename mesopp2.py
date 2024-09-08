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
import redis
import bcrypt
import jwt
import datetime
import uuid

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# JWT secret key
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key')

SESSION_EXPIRY = 3600

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

class AuthState:
    def __init__(self):
        self.user = None
        self.session_id = None

auth_state = AuthState()

def authenticate_user(username, password):
    user_data = redis_client.hgetall(f"user:{username}")
    if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data[b'password']):
        user_id = user_data.get(b'id', str(uuid.uuid4())).decode('utf-8')
        if not user_data.get(b'id'):
            redis_client.hset(f"user:{username}", 'id', user_id)
        return {'id': user_id, 'username': username}
    return None

def create_user(username, password):
    if redis_client.exists(f"user:{username}"):
        return False
    user_id = str(uuid.uuid4())
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    redis_client.hmset(f"user:{username}", {'id': user_id, 'password': hashed_password})
    return True

def generate_token(username):
    payload = {
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def handle_user_message(message, chat_history):
    if not auth_state.user:
        return "Please log in to use the chatbot.", chat_history

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

def create_session(user_id):
    session_id = generate_session_id()
    redis_client.setex(f"session:{session_id}", SESSION_EXPIRY, user_id)
    return session_id

def generate_session_id():
    return str(uuid.uuid4())

@me.component
def LoginForm():
    username = me.state("")
    password = me.state("")

    def handle_login():
        user = authenticate_user(username.get(), password.get())
        if user:
            session_id = create_session(user['id'])
            auth_state.user = user
            auth_state.session_id = session_id
            me.set_cookie("session_id", session_id, max_age=SESSION_EXPIRY)
            me.notify("Login successful", type="success")
            me.navigate("/chatbot")
        else:
            me.notify("Invalid username or password", type="error")

    def handle_register():
        if create_user(username.get(), password.get()):
            me.notify("User registered successfully", type="success")
        else:
            me.notify("Username already exists", type="error")

    return [
        me.input(placeholder="Username", on_input=lambda e: username.set(e.value)),
        me.input(placeholder="Password", type="password", on_input=lambda e: password.set(e.value)),
        me.button("Login", on_click=handle_login),
        me.button("Register", on_click=handle_register)
    ]

def login():
    username = me.state("Username")
    password = me.state("Password")
    user = authenticate_user(username, password)
    if user:
        auth_state.user = user
        me.notify("Login successful", type="success")
        me.update()
    else:
        me.notify("Invalid username or password", type="error")

def register():
    username = me.state("Username")
    password = me.get_value("Password")
    if create_user(username, password):
        me.notify("User registered successfully", type="success")
    else:
        me.notify("Username already exists", type="error")

def logout():
    if auth_state.session_id:
        redis_client.delete(f"session:{auth_state.session_id}")
    auth_state.user = None
    auth_state.session_id = None
    me.delete_cookie("session_id")
    me.navigate("/")

def get_user_from_session(session_id):
    user_id = redis_client.get(f"session:{session_id}")
    if user_id:
        redis_client.expire(f"session:{session_id}", SESSION_EXPIRY)  # Refresh expiry
        return user_id.decode('utf-8')
    return None

@me.page(path="/", title="Login")
def login_page():
    session_id = me.get_cookie("session_id")
    if session_id:
        user_id = get_user_from_session(session_id)
        if user_id:
            auth_state.user = {'id': user_id, 'username': 'User'}  # You might want to fetch the username from Redis
            auth_state.session_id = session_id
            return [
                me.text(f"Welcome back, {auth_state.user['username']}!"),
                me.button("Logout", on_click=logout),
                me.button("Go to Chatbot", on_click=lambda: me.navigate("/chatbot"))
            ]
    return [
        me.text("Welcome to the UNT Chatbot"),
        LoginForm()
    ]

@me.page(path="/chatbot", title="UNT Chatbot")
def chatbot_page():
    session_id = me.get_cookie("session_id")
    if not session_id or not get_user_from_session(session_id):
        me.navigate("/")
        return []
    return mel.chat(transform, title="UNT Chatbot", bot_user="Chatbot")

def transform(user_input: str, history: list) -> str:
    if not auth_state.user:
        return "Please log in to use the chatbot."
    
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