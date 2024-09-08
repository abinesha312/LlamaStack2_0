import mesop as me
from .auth import Auth

@me.page(path="/register", title="Register")
def register_page():
    username = me.input(label="Username")
    password = me.input(label="Password", type="password")
    confirm_password = me.input(label="Confirm Password", type="password")
    
    if me.button("Register"):
        if password.value != confirm_password.value:
            me.error("Passwords do not match")
        else:
            success, message = Auth.register(username.value, password.value)
            if success:
                me.success(message)
                me.navigate("/login")
            else:
                me.error(message)