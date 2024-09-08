import hashlib
import uuid
from .redis_client import redis_client

class Auth:
    @staticmethod
    def register(username, password):
        if redis_client.exists(username):
            return False, "Username already exists"
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        redis_client.set(username, hashed_password)
        return True, "Registration successful"

    @staticmethod
    def login(username, password):
        stored_password = redis_client.get(username)
        if stored_password is None:
            return False, "User not found", None
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if hashed_password == stored_password.decode():
            session_id = str(uuid.uuid4())
            redis_client.set(f"session:{session_id}", username, ex=3600)  # Session expires in 1 hour
            return True, "Login successful", session_id
        return False, "Incorrect password", None

    @staticmethod
    def logout(session_id):
        redis_client.delete(f"session:{session_id}")