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
import hashlib
import uuid
import redis
from AuthFLD.Auth import Auth
from AuthFLD.models import UserDetails

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DB_FAISS_PATH = 'vectorstore/db_faiss'
redis_client = redis.Redis(host='localhost', port=6379, db=0)

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
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['context', 'question'])

def load_llm():
    model_path = '/home/haridoss/Models/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'
    try:
        hf_pipeline = pipeline('text-generation', model=model_path, device=0)
        return HuggingFacePipeline(pipeline=hf_pipeline)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def create_chatbot_chain(llm, prompt, db, memory):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 10}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

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
    print(qa)
    response = qa({'question': message})
    answer_marker = "Helpful answer:\n"
    answer_start = response['answer'].find(answer_marker)
    helpful_answer = response['answer'][answer_start + len(answer_marker):].strip() if answer_start != -1 else "Helpful answer not found."

    chat_history.append(ChatMessage(role="user", content=message))
    chat_history.append(ChatMessage(role="assistant", content=helpful_answer))
    
    return "", chat_history

@me.stateclass
class AppState:
    session_id: str = ""
    user_id: str = ""
    username: str = ""
    error_message: str = ""

@me.stateclass
class RegisterState:
    username: str = ""
    password: str = ""
    confirm_password: str = ""
    name: str = ""
    email: str = ""
    contact_number: str = ""
    error_message: str = ""

def on_blur_UN(e: me.InputBlurEvent):
    state = me.state(AppState)
    state.username = e.value

def on_blur_Pass(e: me.InputBlurEvent):
    state = me.state(AppState)
    state.password = e.value

def on_blur_UN_Rgs(e: me.InputBlurEvent):
    state = me.state(RegisterState)
    state.username = e.value

def on_blur_Upass_Rgs(e: me.InputBlurEvent):
    state = me.state(RegisterState)
    state.password = e.value
    state.Cnfrmpassword = e.value

def is_authenticated():
    state = me.state(AppState)
    print("Is Auth function", state)
    return bool(state.session_id)

@me.page(path="/", title="UNT Chatbot")
def main_page():
    if not is_authenticated():
        me.navigate("/login")
    else:
        chatbot_page()

@me.stateclass
class LoginState:
    username: str = ""
    password: str = ""
    error_message: str = ""
    
@me.stateclass
class LoginState:
    username: str = ""
    password: str = ""
    error_message: str = ""


@me.page(
    path="/login",
    title="Login",
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io", "https://huggingface.co"]
    ),
)
def login_page():
    me.set_theme_mode("dark")
    state = me.state(LoginState)
    app_state = me.state(AppState)

    with me.box(style=me.Style(padding=me.Padding.all(30))):
        me.text("Login", type="headline-4")

        with me.box(style=me.Style(margin=me.Margin(top=20))):
            me.input(
                label="Username",
                on_blur=lambda e: setattr(state, 'username', e.value),
                style=me.Style(width="100%")
            )

        with me.box(style=me.Style(margin=me.Margin(top=15))):
            me.input(
                label="Password",
                type="password",
                on_blur=lambda e: setattr(state, 'password', e.value),
                style=me.Style(width="100%")
            )

        with me.box(style=me.Style(margin=me.Margin(top=20))):
            me.button(
                "Login",
                on_click=handle_login,
                color="primary",
                type="raised"
            )

        if state.error_message:
            with me.box(style=me.Style(margin=me.Margin(top=10))):
                me.text(
                    state.error_message,
                    style=me.Style(color="red")
                )

        with me.box(style=me.Style(margin=me.Margin(top=15))):
            me.button(
                "Register",
                on_click=lambda _: me.navigate("/register"),
                type="flat"
            )

    def handle_login(e: me.ClickEvent):
        state = me.state(LoginState)
        app_state = me.state(AppState)
        
        success, message, session_id, user_id = Auth.login(redis_client, state.username, state.password)
        if success:
            app_state.session_id = session_id
            app_state.user_id = user_id
            app_state.username = state.username
            me.navigate("/chatbot")
        else:
            state.error_message = message
@me.stateclass
class RegisterState:
    username: str = ""
    password: str = ""
    confirm_password: str = ""
    name: str = ""
    email: str = ""
    contact_number: str = ""
    error_message: str = ""

@me.page(path="/register", title="Register")
def register_page():
    state = me.state(RegisterState)

    me.text("Register", type="headline-4")

    me.input(label="Username", on_blur=lambda e: setattr(state, 'username', e.value))
    me.input(label="Password", type="password", on_blur=lambda e: setattr(state, 'password', e.value))
    me.input(label="Confirm Password", type="password", on_blur=lambda e: setattr(state, 'confirm_password', e.value))
    me.input(label="Name", on_blur=lambda e: setattr(state, 'name', e.value))
    me.input(label="Email", on_blur=lambda e: setattr(state, 'email', e.value))
    me.input(label="Contact Number", on_blur=lambda e: setattr(state, 'contact_number', e.value))

    def handle_register(e: me.ClickEvent):
        if state.password != state.confirm_password:
            state.error_message = "Passwords do not match"
        else:
            success, message = Auth.register(redis_client, state.username, state.password, state.name, state.email, state.contact_number)
            if success:
                state.error_message = ""
                me.navigate("/login")
            else:
                state.error_message = message

    me.button("Register", on_click=handle_register, color="primary", type="raised")
    
    if state.error_message:
        me.text(state.error_message, style=me.Style(color="red"))

    me.button("Back to Login", on_click=lambda _: me.navigate("/login"), type="flat")

@me.page(path="/logout", title="Logout")
def logout_page():
    state = me.state(AppState)
    if state.session_id:
        Auth.logout(state.session_id)
    state.session_id = ""
    state.user_id = ""
    state.username = ""
    me.navigate("/login")
    Auth.logout(redis_client, state.session_id)

@me.page(path="/chatbot", title="UNT Chatbot")
def chatbot_page():
    if not is_authenticated():
        me.navigate("/login")
    else:
        state = me.state(AppState)
        me.text(f"Welcome, {state.username} !", type="headline-5")
        mel.chat(transform, title="UNT Chatbot", bot_user="Chatbot")
        me.button("Logout", on_click=lambda _: me.navigate("/logout"), type="flat")

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
    print("Model loaded successfully")
    me.run()

if __name__ == "__main__":
    main()