import uuid
import hashlib

class UserDetails:
    def __init__(self, username, password, name, email, contact_number):
        self.user_id = str(uuid.uuid4())
        self.username = username
        self.password = hashlib.sha256(password.encode()).hexdigest()
        self.name = name
        self.email = email
        self.contact_number = contact_number

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "password": self.password,
            "name": self.name,
            "email": self.email,
            "contact_number": self.contact_number
        }

    @classmethod
    def from_dict(cls, data):
        user = cls(
            username=data['username'],
            password=data['password'],
            name=data['name'],
            email=data['email'],
            contact_number=data['contact_number']
        )
        user.user_id = data['user_id']
        return user