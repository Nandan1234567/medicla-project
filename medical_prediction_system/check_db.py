from pymongo import MongoClient
import pprint

def check_db():
    try:
        # Connect
        client = MongoClient('mongodb://localhost:27017/')

        # Check if database exists
        if 'medical_healthcare' not in client.list_database_names():
            print("WARNING: Database 'medical_healthcare' does not exist yet. Register a user to create it.")
            return

        db = client['medical_healthcare']
        print("=== ✅ Database Connection ESTABLISHED ===")
        print(f"📂 Database Name: {db.name}")

        # Check Users
        user_count = db.users.count_documents({})
        users = list(db.users.find({}, {'_id': 0, 'password': 0, 'api_token': 0}))
        print(f"\n=== 👤 Users Found: {user_count} ===")
        for u in users:
            print(f" - Name: {u.get('name')}, Email: {u.get('email')}, Provider: {u.get('provider', 'local')}")

        # Check Predictions
        pred_count = db.predictions.count_documents({})
        predictions = list(db.predictions.find({}, {'_id': 0}).sort('timestamp', -1).limit(5))
        print(f"\n=== 🩺 Predictions Found: {pred_count} (Showing last 5) ===")

        if predictions:
            print("--- Raw Sample Record ---")
            pprint.pprint(predictions[0])
            print("-------------------------")

        for p in predictions:
            # Flexible printing based on potential keys
            disease = p.get('disease') or p.get('prediction') # Fallback key
            severity = p.get('severity') or p.get('urgencyLevel') # Fallback key
            print(f" - Disease: {disease} | Severity: {severity}")

    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    check_db()
