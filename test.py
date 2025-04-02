
import time
import pickle
import threading
import queue
import numpy as np
import pandas as pd
import joblib
from scapy.all import sniff, IP, TCP, UDP

ordered_features = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", 
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", 
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", 
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty"
]


categorical_columns = ["protocol_type", "service", "flag"]


default_feature_values = {
    "duration": 0,
    "protocol_type": "other",
    "service": "other",
    "flag": "none",
    "src_bytes": 0,
    "dst_bytes": 0,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 0,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 0,
    "srv_count": 0,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 0.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 0,
    "dst_host_srv_count": 0,
    "dst_host_same_srv_rate": 0.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
    "attack": "normal",
    "difficulty": 0
}


try:
    with open('onehot_encoder.pkl', 'rb') as f:
        encoder = joblib.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)
    with open('selector.pkl', 'rb') as f:
        selector = joblib.load(f)
    with open('Random_Forest_model.pkl', 'rb') as f:
        model = joblib.load(f)
except Exception as e:
    print("Error loading pre-trained objects:", e)
    exit(1)


def infer_service(sport, dport):
    common_ports = {80: 'http', 443: 'https', 21: 'ftp', 22: 'ssh', 25: 'smtp', 110: 'pop3'}
    if sport in common_ports:
        return common_ports[sport]
    elif dport in common_ports:
        return common_ports[dport]
    else:
        return "other"


def extract_features(packet):
    features = default_feature_values.copy()
    try:
        
        features["duration"] = 0

       
        if packet.haslayer(IP):
            proto_num = packet[IP].proto
            protocol_mapping = {6: 'tcp', 17: 'udp'}
            features["protocol_type"] = protocol_mapping.get(proto_num, "other")
        # Service and flag: derived from TCP/UDP layers.
        if packet.haslayer(TCP):
            sport = packet[TCP].sport
            dport = packet[TCP].dport
            features["service"] = infer_service(sport, dport)
            features["flag"] = str(packet[TCP].flags)
        elif packet.haslayer(UDP):
            sport = packet[UDP].sport
            dport = packet[UDP].dport
            features["service"] = infer_service(sport, dport)
            features["flag"] = "none"
        pkt_len = len(packet)
        features["src_bytes"] = pkt_len
        features["dst_bytes"] = pkt_len
    except Exception as e:
        print("Error extracting packet features:", e)
    return features


def preprocess_features(features_dict):
    feature_columns = [col for col in ordered_features if col not in ["attack", "difficulty"]]
    data_row = {col: features_dict.get(col, default_feature_values[col]) for col in feature_columns}
    df = pd.DataFrame([data_row], columns=feature_columns)

    try:
        X_cat = encoder.transform(df[categorical_columns])
    except Exception as e:
        print("Error during categorical transformation:", e)
        X_cat = np.zeros((df.shape[0], len(encoder.get_feature_names_out(categorical_columns))))
    

    X_num = df.drop(columns=categorical_columns).values

   
    X_transformed = np.hstack([X_num, X_cat])
    
   
    try:
        X_scaled = scaler.transform(X_transformed)
    except Exception as e:
        print("Error during scaling transformation:", e)
        X_scaled = X_transformed
    
   
    try:
        X_selected = selector.transform(X_scaled)
    except Exception as e:
        print("Error during feature selection transformation:", e)
        X_selected = X_scaled
    return X_selected


packet_queue = queue.Queue()


def processing_worker():
    while True:
        try:
            features = packet_queue.get(timeout=1)
            processed_features = preprocess_features(features)
            prediction = model.predict(processed_features)
            print(f"Predicted: {prediction[0]} | Features: {features}")
            packet_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print("Error in processing worker:", e)


def main():
    print("Starting single packet capture loop... (Press CTRL+C to stop)")
    while True:
       
        packets = sniff(count=1, timeout=5,iface = "Wi-Fi")
        if packets:
            packet = packets[0]
            features = extract_features(packet)
            processed_features = preprocess_features(features)
            prediction = model.predict(processed_features)
            print(f"Predicted: {prediction[0]} | Features: {features}")
        else:
            print("No packet captured.")
        
        time.sleep(2) 

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
