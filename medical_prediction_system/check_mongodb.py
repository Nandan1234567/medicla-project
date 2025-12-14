"""
Check MongoDB Connection Status
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

def check_mongodb_connection():
    """Check if MongoDB is connected and working"""

    print("\n" + "="*70)
    print("MONGODB CONNECTION TEST")
    print("="*70 + "\n")

    try:
        # Try to connect
        print("1. Attempting to connect to MongoDB...")
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)

        # Test connection
        print("2. Testing connection...")
        client.admin.command('ping')
        print("   ✅ MongoDB is CONNECTED and RESPONDING\n")

        # Check database
        db = client['medical_healthcare']
        print("3. Checking database 'medical_healthcare'...")
        collections = db.list_collection_names()
        print(f"   ✅ Database exists with collections: {collections}\n")

        # Check users
        user_count = db.users.count_documents({})
        print(f"4. Users in database: {user_count}")
        if user_count > 0:
            print("   ✅ User data is being stored\n")
        else:
            print("   ⚠️  No users yet. Register to test storage.\n")

        # Check predictions
        pred_count = db.predictions.count_documents({})
        print(f"5. Predictions in database: {pred_count}")
        if pred_count > 0:
            print("   ✅ Prediction data is being stored\n")
        else:
            print("   ⚠️  No predictions yet. Make a prediction to test storage.\n")

        print("="*70)
        print("RESULT: MongoDB is CONNECTED and WORKING! ✅")
        print("="*70 + "\n")

        return True

    except ServerSelectionTimeoutError:
        print("   ❌ FAILED: Cannot connect to MongoDB\n")
        print("TROUBLESHOOTING:")
        print("1. Make sure MongoDB is running:")
        print("   - Double-click 'start_mongodb.bat'")
        print("   - Keep the window open\n")
        print("2. Check if port 27017 is available")
        print("3. Restart MongoDB and try again\n")
        return False

    except ConnectionFailure as e:
        print(f"   ❌ FAILED: Connection error: {e}\n")
        return False

    except Exception as e:
        print(f"   ❌ FAILED: Unexpected error: {e}\n")
        return False

if __name__ == "__main__":
    check_mongodb_connection()
