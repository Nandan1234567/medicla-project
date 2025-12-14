# MongoDB Quick Start for Windows

## Option 1: Start MongoDB Service (If Installed as Service)

Open **PowerShell as Administrator** and run:
```powershell
net start MongoDB
```

## Option 2: Start MongoDB Manually

### Step 1: Find MongoDB Installation
Common locations:
- `C:\Program Files\MongoDB\Server\7.0\bin\mongod.exe`
- `C:\Program Files\MongoDB\Server\6.0\bin\mongod.exe`
- `C:\Program Files\MongoDB\Server\5.0\bin\mongod.exe`

### Step 2: Create Data Directory
```powershell
mkdir C:\data\db
```

### Step 3: Start MongoDB
Open a **new PowerShell window** and run:
```powershell
# Replace with your actual MongoDB path
& "C:\Program Files\MongoDB\Server\7.0\bin\mongod.exe" --dbpath C:\data\db
```

**Keep this window open!** MongoDB needs to run in the background.

### Step 4: Verify Connection
Open **another PowerShell window** and run:
```powershell
& "C:\Program Files\MongoDB\Server\7.0\bin\mongosh.exe"
```

If you see the MongoDB shell, it's working! Type `exit` to close.

## Option 3: Use MongoDB Compass (GUI)

If you installed MongoDB Compass:
1. Open **MongoDB Compass**
2. Connect to: `mongodb://localhost:27017`
3. You should see the connection succeed

## Quick Test Script

Save this as `start_mongodb.bat` in your project folder:
```batch
@echo off
echo Starting MongoDB...
"C:\Program Files\MongoDB\Server\7.0\bin\mongod.exe" --dbpath C:\data\db
```

Then just double-click it to start MongoDB!

## Restart Flask App

After MongoDB is running:
1. Stop your Flask server (Ctrl+C)
2. Start it again:
```bash
python backend_flask/app.py
```

You should see: `MongoDB connected successfully`

## Troubleshooting

**Can't find mongod.exe?**
- Search your computer for "mongod.exe"
- Or reinstall MongoDB from: https://www.mongodb.com/try/download/community

**Port already in use?**
- Another MongoDB instance might be running
- Check Task Manager and close any mongod.exe processes

**Permission denied?**
- Run PowerShell as Administrator
- Make sure C:\data\db folder exists
