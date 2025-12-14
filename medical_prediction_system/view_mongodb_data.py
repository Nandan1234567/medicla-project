"""
View MongoDB Data - Medical Prediction System
This script shows all data stored in MongoDB
"""

from pymongo import MongoClient
from datetime import datetime
import json

def view_mongodb_data():
    """Connect to MongoDB and display all stored data"""

    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['medical_healthcare']

        print("\n" + "="*70)
        print("MONGODB DATA VIEWER - Medical Prediction System")
        print("="*70)

        # Check connection
        client.admin.command('ping')
        print("✓ Connected to MongoDB successfully!\n")

        # View Users Collection
        print("-"*70)
        print("USERS COLLECTION")
        print("-"*70)
        users = list(db.users.find())

        if users:
            print(f"Total Users: {len(users)}\n")
            for i, user in enumerate(users, 1):
                print(f"User #{i}:")
                print(f"  Name: {user.get('name', 'N/A')}")
                print(f"  Email: {user.get('email', 'N/A')}")
                print(f"  Phone: {user.get('phone', 'N/A')}")
                print(f"  Gender: {user.get('gender', 'N/A')}")
                print(f"  Registered: {user.get('createdAt', 'N/A')}")
                print(f"  Last Login: {user.get('lastLogin', 'Never')}")
                print(f"  Active: {user.get('isActive', True)}")
                print()
        else:
            print("No users found. Register a user to see data here.\n")

        # View Predictions Collection
        print("-"*70)
        print("PREDICTIONS COLLECTION")
        print("-"*70)
        predictions = list(db.predictions.find())

        if predictions:
            print(f"Total Predictions: {len(predictions)}\n")
            for i, pred in enumerate(predictions, 1):
                print(f"Prediction #{i}:")
                print(f"  Symptoms: {pred.get('symptoms', 'N/A')}")
                print(f"  Disease: {pred.get('result', {}).get('disease', 'N/A')}")
                print(f"  Confidence: {pred.get('result', {}).get('confidence', 0)}%")
                print(f"  Emergency: {pred.get('isEmergency', False)}")
                print(f"  Date: {pred.get('createdAt', 'N/A')}")
                print()
        else:
            print("No predictions found. Make a prediction to see data here.\n")

        # Database Statistics
        print("-"*70)
        print("DATABASE STATISTICS")
        print("-"*70)
        print(f"Database Name: medical_healthcare")
        print(f"Collections: {db.list_collection_names()}")
        print(f"Total Users: {db.users.count_documents({})}")
        print(f"Total Predictions: {db.predictions.count_documents({})}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error connecting to MongoDB: {e}")
        print("\nMake sure:")
        print("1. MongoDB is running (start_mongodb.bat)")
        print("2. MongoDB is on port 27017")
        print("\n")

if __name__ == "__main__":
    view_mongodb_data()
