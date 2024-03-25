from pymongo import MongoClient

client = MongoClient(
    "mongodb+srv://phanphanvjc:kFw0Y6ojRotOUHN6@cluster0.dhkmjog.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client.chatbot

PROMPT_COLLECTION = db['prompts']
PARAMETER_COLLECTION = db['parameters']