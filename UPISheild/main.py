# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import re
import cv2
import pandas as pd
import os
from datetime import datetime
import easyocr
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
import streamlit as st
from PIL import Image

# %%
reader = easyocr.Reader(['en'])
sia = SentimentIntensityAnalyzer()


# %%
# -------- Extract UPI ID from link ----------
def extract_upi_id_from_link(link):
    match = re.search(r'pa=([^&]+)', link)
    return match.group(1) if match else None

# -------- Read QR Code and extract UPI ID ----------
def extract_qr_data(image_path):
    detector = cv2.QRCodeDetector()
    img = cv2.imread(image_path)
    data, bbox, _ = detector.detectAndDecode(img)
    return data

# -------- Unified input handler ----------
# def get_upi_input():
#     print("Select Input Type:")
#     print("1. Enter UPI ID manually")
#     print("2. Enter UPI payment link")
#     print("3. Upload QR code image")

#     choice = input("Enter choice (1/2/3): ").strip()

#     if choice == '1':
#         upi_id = input("Enter UPI ID (e.g., abc@upi): ").strip()
#     elif choice == '2':
#         link = input("Paste UPI link (e.g., upi://pay?...): ").strip()
#         upi_id = extract_upi_id_from_link(link)
#         if not upi_id:
#             print("Could not extract UPI ID from link.")
#             return None
#     elif choice == '3':
#         path = input("Enter image path (e.g., ./qr.png): ").strip()
#         upi_id = extract_qr_data(path)
#         if not upi_id:
#             print("No valid QR code found.")
#             return None
#     else:
#         print("Invalid choice.")
#         return None

#     print(f"\n✅ Extracted UPI ID: {upi_id}")
#     return upi_id

def get_upi_input():
    st.subheader("Select Input Type")
    choice = st.radio("Choose input method:", ["Enter UPI ID manually", "Paste UPI link", "Upload QR code image"])

    upi_id = None

    if choice == "Enter UPI ID manually":
        upi_id = st.text_input("Enter UPI ID (e.g., abc@upi)")

    elif choice == "Paste UPI link":
        link = st.text_input("Paste UPI payment link (e.g., upi://pay?...):")
        if link:
            upi_id = extract_upi_id_from_link(link)
            if not upi_id:
                st.error("❌ Could not extract UPI ID from link.")
            else:
                st.success(f"✅ Extracted UPI ID: {upi_id}")

    elif choice == "Upload QR code image":
        file = st.file_uploader("Upload QR image", type=['png', 'jpg', 'jpeg'])
        if file is not None:
            img_path = f"temp_qr.png"
            with open(img_path, "wb") as f:
                f.write(file.getbuffer())
            upi_id = extract_qr_data(img_path)
            if not upi_id:
                st.error("❌ No valid QR code found.")
            else:
                st.success(f"✅ Extracted UPI ID: {upi_id}")

    return upi_id



# %%
#extract features

def extract_username(handle):
    return handle.partition('@')[0]
def extract_domain(handle):
    return handle.partition('@')[2]
def detect_digits(handle):
    for i in handle:
        if(i.isdigit()):
            return 1;
    return 0;

def extract_features(df):
    df['USERNAME']=df['UPI'].apply(extract_username)
    df['DOMAIN']=df['UPI'].apply(extract_domain)
    df['HANDLE_LENGTH'] = df['UPI'].apply(len)
    df['HAS_DIGITS'] = df['USERNAME'].apply(detect_digits)
    scam_keywords = ['refund', 'loan','cash' , 'reward', 'verify']
    df['HAS_KEYWORDS'] = df['USERNAME'].apply(lambda x: any(k in x.lower() for k in scam_keywords))


# %%
def predict_anomaly(upi_id, pipeline):
    file=pd.read_csv('upi_anomaly_dataset.csv')
    row = file[file['UPI'] == upi_id]

    if not row.empty:
        reports = row['REPORTS'].values[0]
    else:
        if os.path.exists('community_reports.csv'):
            file2 = pd.read_csv('community_reports.csv')
            reports=(file2['upi_id'] == upi_id).sum()
        else:
            reports=0

    df=pd.DataFrame([{
        'UPI':upi_id,
        'REPORTS':reports
    }])
    extract_features(df)
    
    features = ['REPORTS', 'DOMAIN', 'HANDLE_LENGTH', 'HAS_DIGITS', 'HAS_KEYWORDS']
    prediction = pipeline.predict(df[features])[0]
    return prediction


# %%
def extract_text_from_image(image_path):
    """
    Extracts visible text from an image using EasyOCR.
    """
    results = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(results)
    return extracted_text


# %%
SCAM_KEYWORDS = [
    # Urgency / Threat
    "verify", "account blocked", "urgent", "limited time", "immediate action", "final warning",
    "update now", "act fast", "deadline", "security alert", "deactivation", "unauthorized access",

    # Refund / Payment Scams
    "refund", "payment failed", "transaction failed", "pay immediately", "receive money",
    "get back", "pending amount", "recharge issue", "loan approved", "claim refund", "wrong payment",

    # Clickbait / Phishing Triggers
    "click here", "open link", "tap now", "scan to receive", "visit now", "login here", "redeem now",
    "login quickly", "scan QR", "follow instructions",

    # Free / Prize / Reward
    "win", "reward", "free", "gift", "bonus", "offer ends", "cashback", "you are selected", 
    "lucky draw", "free recharge", "prize claim",

    # OTP / Fake Verification
    "OTP", "activation", "your account", "bank issue", "validate", "confirm your identity",
    "update KYC", "kyc expired", "complete KYC", "aadhaar verify", "PAN verification", "reset password",

    # UPI/BharatPay Specific
    "UPI refund", "UPI support", "BharatPay", "Scan this code", "QR payment", "check transaction",
    "pay ₹1 to verify", "request sent", "account credited",

    # Psychological pressure
    "emergency", "family issue", "accident", "hospital", "doctor fee", "help me", "need money urgently",
    "money stuck", "I’m stranded", "mom in hospital", "ambulance"

]


# %%
def context_score(text,SCAM_KEYWORDS):
    if not text:
        return {
            'text': '',
            'scam_keywords_count': 0,
            'sentiment': {},
            'context_score': 0.0,
            'suspicious': False
        }

    # Normalize
    text_lower = text.lower()

    # Keyword scam score
    scam_hits = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    scam_score = len(scam_hits)

    # Sentiment
    sentiment = sia.polarity_scores(text)

    # Combine scores
    context_score = (scam_score * 0.5) + (sentiment['neg'] * 5)  # Weight keywords + negative tone
    suspicious = context_score > 2  

    return {
        'text': text,
        'scam_keywords': scam_hits,
        'scam_keywords_count': scam_score,
        'sentiment': sentiment,
        'context_score': round(context_score, 2),
        'suspicious': suspicious
    }


# %%
def explain_prediction(upi_id, context_result, anomaly_score):
    if context_result['suspicious'] and anomaly_score == -1:
        reason = f"The system flagged this as suspicious because the message contains keywords like {context_result['scam_keywords']} and had a high negative sentiment."
    elif context_result['suspicious']:
        reason = f"The context around this message raised red flags due to keywords like {context_result['scam_keywords']} and emotional tone."
    elif anomaly_score == -1:
        reason = f"The UPI ID '{upi_id}' shows unusual behavior compared to common patterns, based on our anomaly detection model."
    else:
        reason = f"The UPI ID '{upi_id}' appears normal based on both its behavior and message context."
    
    return reason


# %%
def mini_bot(user_input, last_result):
    user_input = user_input.lower()

    if "why" in user_input and "flag" in user_input:
        return explain_prediction(last_result['upi_id'], last_result['context'], last_result['model_pred'])
    elif "what is scam score" in user_input or "context score" in user_input:
        return f"The context score combines scam keywords and negative sentiment. A higher score means more likely scam intent."
    elif "how was this helpful" in user_input:
        return "This explanation is based on behavioral anomaly + message content. We aim to help you understand risk clearly."
    else:
        return "I’m a mini helper bot! Try asking 'Why was this flagged?' or 'What is context score?'"


# %%
def community_report(upi_id, message, reason, reporter=None):
    """
    Stores community-reported scam data for future scoring/tracking.
    """
    report = {
        "upi_id": upi_id,
        "message": message,
        "reason": reason,
        "reporter": reporter if reporter else "anonymous",
        "timestamp": datetime.now().isoformat()
    }

    file_path = "community_reports.csv"

    # Append or create
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([report])], ignore_index=True)
    else:
        df = pd.DataFrame([report])

    df.to_csv(file_path, index=False)
    print("✅ Report saved. Thank you for contributing!")

    return True


# %%
def main():
    st.title("UPIShield: UPI Fraud Detection & Context Scoring")

    upi_id=get_upi_input()
    uploaded_file = st.file_uploader("Upload UPI Screenshot Image", type=["png","jpg","jpeg"])
    if upi_id and uploaded_file:
        
        #Anomaly scoring
        pipeline=joblib.load('upi_anomaly_pipeline.pkl')
        anomaly_pred = predict_anomaly(upi_id,pipeline)
        st.markdown(f"### Anomaly Prediction for `{upi_id}`:")
        st.write("⚠️ Suspicious" if anomaly_pred == -1 else "✅ Normal")

        # Context scoring
        img_path = f"temp_ss.png"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        extracted_text=extract_text_from_image(img_path)
        context_result = context_score(extracted_text,SCAM_KEYWORDS)
        st.markdown("### Context Score:")
        st.write(context_result['context_score'])

        # Explaination
        explanation = explain_prediction(upi_id, context_result, anomaly_pred)
        st.info(explanation)

        # Mini bot chat interface
        st.subheader("Ask UPIShield Bot 👇")
        user_question = st.text_input("Ask me why this was flagged or about the context score")

        if user_question:
            bot_reply = mini_bot(user_question, {
                'upi_id': upi_id,
                'context': context_result,
                'model_pred': anomaly_pred
            })
            st.success(bot_reply)

        # Community reporting form
        with st.expander("🛡️ Report this UPI/message as suspicious"):
            reporter = st.text_input("Your name (optional)", placeholder="Leave blank for anonymous")
            reason = st.text_area("Reason for reporting this as suspicious")
            if st.button("Report Now"):
                community_report(upi_id, extracted_text, reason, reporter or "anonymous")
                st.success("Thank you! Your report is saved.")

        # Feedback on explanation
        st.subheader("Was this explanation helpful?")
        feedback = st.radio("", ["Yes", "No"])
        if st.button("Submit Feedback"):
            # Here, you would save feedback in DB or file
            st.success("Thanks for your feedback! It helps us improve.")

if __name__ == "__main__":
    main()

# %%
