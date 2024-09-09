import hashlib
import uuid

class Auth:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    @staticmethod
    def register(redis_client, username, password, name, email, contact_number):
        if redis_client.exists(f"user:{username}"):
            return False, "Username already exists"
        user_id = str(uuid.uuid4())
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        user_data = {
            "user_id": user_id,
            "username": username,
            "password": hashed_password,
            "name": name,
            "email": email,
            "contact_number": contact_number
        }
        redis_client.hmset(f"user:{username}", user_data)
        return True, "Registration successful"

    @staticmethod
    def login(redis_client, username, password):
        user_data = redis_client.hgetall(f"user:{username}")
        if not user_data:
            return False, "User not found", None, None
        
        user_dict = {k.decode(): v.decode() for k, v in user_data.items()}
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        if hashed_password == user_dict['password']:
            session_id = str(uuid.uuid4())
            redis_client.setex(f"session:{session_id}", 3600, user_dict['user_id'])  # Session expires in 1 hour
            return True, "Login successful", session_id, user_dict['user_id']
        return False, "Incorrect password", None, None

    @staticmethod
    def logout(redis_client, session_id):
        redis_client.delete(f"session:{session_id}")