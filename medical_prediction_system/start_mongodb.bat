@echo off
echo ========================================
echo Starting MongoDB Server
echo ========================================
echo.

REM Create data directory if it doesn't exist
if not exist "C:\data\db" (
    echo Creating data directory...
    mkdir "C:\data\db"
)

echo Starting MongoDB on port 27017...
echo Keep this window open while using the app!
echo.
echo Press Ctrl+C to stop MongoDB
echo ========================================
echo.

"C:\Program Files\MongoDB\Server\8.2\bin\mongod.exe" --dbpath "C:\data\db"
