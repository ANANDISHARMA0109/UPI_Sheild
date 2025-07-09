# UPI Shield 🛡️

Developed by: ANANDISHARMA0109, anannya-2032

Advanced fraud detection system for UPI transactions combining:
- **Anomaly Detection** (Machine Learning)
- **Context Analysis** (NLP/Sentiment)
- **Community Reporting** (Crowdsourced intelligence)

## Features

✅ **Multi-Input Support**  
   - Manual UPI ID entry
   - Payment link parsing
   - QR code scanning

🔍 **Context Analysis**  
   - 50+ scam keyword detection
   - Sentiment scoring (NLTK)
   - Image text extraction (EasyOCR)

🤖 **Interactive Bot**  
   - Explains risk factors
   - Answers questions about scoring

📊 **Community Reporting**  
   - Crowdsourced scam tracking
   - Persistent storage (CSV)

## Tech Stack

- **Core**: Python 3.9+
- **ML**: Scikit-learn, Joblib
- **NLP**: NLTK, EasyOCR
- **UI**: Streamlit
- **Computer Vision**: OpenCV

## Installation

```bash
git clone https://github.com/ANANDISHARMA0109/UPI_Sheild.git
cd upi-shield
pip install -r requirements.txt
streamlit run main.py
