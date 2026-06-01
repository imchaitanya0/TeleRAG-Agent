#!/bin/bash
# download_data.sh - Script to download necessary data for TeleRAG-Agent

echo "=== TeleRAG-Agent Data Acquisition ==="

# Create directories
mkdir -p data/raw/teleqna
mkdir -p data/raw/3gpp
mkdir -p data/raw/oran

echo ""
echo "[1] TeleQnA Dataset"
echo "Instructions:"
echo "1. Visit https://github.com/netop-team/TeleQnA"
echo "2. Download TeleQnA.zip"
echo "3. Extract using password: teleqnadataset into data/raw/teleqna/"
echo "(Automated download skipped due to authentication/password requirements)"

echo ""
echo "[2] NetsLab-5GORAN-IDD Dataset"
echo "Instructions:"
echo "1. Ensure you have the Kaggle API installed (pip install kaggle)"
echo "2. Configure your ~/.kaggle/kaggle.json credentials"
echo "3. Run: kaggle datasets download -d netslabdemo/netslab-5g-oran-idd -p data/raw/oran/ --unzip"
echo "Note: This is real O-RAN testbed data containing telemetry and attack scenarios."

echo ""
echo "[3] 3GPP Specifications"
echo "Run the python script to download curated 3GPP PDFs:"
echo "python scripts/download_3gpp_specs.py"

echo ""
echo "Data acquisition script complete. Please follow manual steps if needed."
