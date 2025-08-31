echo "Starting Mathematical Expression Converter Web Interface..."
echo "========================================================="


if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "checkpoint.ckpt" ]; then
    echo "Error: checkpoint.ckpt not found in current directory"
    exit 1
fi

if [ ! -f "dictionary.txt" ]; then
    echo "Error: dictionary.txt not found in current directory"
    exit 1
fi

if [ ! -d "static" ]; then
    echo "Error: static directory not found"
    exit 1
fi

echo "Checking dependencies..."
python3 -c "import fastapi, torch, cv2, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi


python3 app.py
