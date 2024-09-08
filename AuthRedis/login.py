import mesop as me
from .auth import Auth

@me.stateclass
class LoginState:
    error_message: str = ""
    username: str = ""
    password: str = ""

@me.page(path="/login", title="Login")
def login_page():
    state = me.state(LoginState)

    me.text("Login", type="headline-4")
    
    username = me.input(label="Username")
    password = me.input(label="Password", type="password")

    def handle_login(e: me.ClickEvent):
        state.username = username.value
        state.password = password.value

        if not state.username or not state.password:
            state.error_message = "Username or password is empty"
            return

        success, message, session_id = Auth.login(state.username, state.password)
        if success:
            me.navigate("/chatbot")
        else:
            state.error_message = message

    me.button("Login", on_click=handle_login)
    
    if state.error_message:
        me.text(state.error_message, style=me.Style(color="red"))

    # Debug information
    me.text(f"Current username: {state.username}")
    me.text(f"Current password: {state.password}")