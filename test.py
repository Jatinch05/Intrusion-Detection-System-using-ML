import time
import pickle
import threading
import queue
import numpy as np
import pandas as pd
import joblib
from scapy.all import sniff, IP, TCP, UDP, ICMP
from collections import defaultdict, deque
from datetime import datetime, timedelta
import socket

# Feature definitions
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

# Connection tracking structures
connection_states = {}  # Track connection states
connection_history = deque(maxlen=10000)  # Keep last 10000 connections
host_statistics = defaultdict(lambda: {
    'connections': deque(maxlen=1000),
    'services': defaultdict(int),
    'errors': defaultdict(int),
    'same_srv_connections': deque(maxlen=1000),
    'src_ports': set()
})

# Time window for statistical calculations (2 seconds)
TIME_WINDOW = 2

class ConnectionRecord:
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol, service):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.service = service
        self.start_time = time.time()
        self.end_time = None
        self.src_bytes = 0
        self.dst_bytes = 0
        self.flag = "SF"  # Default to successful
        self.urgent = 0
        self.hot = 0
        self.num_failed_logins = 0
        self.logged_in = 0
        self.num_compromised = 0
        self.root_shell = 0
        self.su_attempted = 0
        self.num_root = 0
        self.num_file_creations = 0
        self.num_shells = 0
        self.num_access_files = 0
        self.num_outbound_cmds = 0
        self.is_host_login = 0
        self.is_guest_login = 0
        self.land = 0
        self.wrong_fragment = 0

def get_service_from_port(port, protocol):
    """Map port numbers to service names"""
    tcp_services = {
        7: 'echo', 9: 'discard', 13: 'daytime', 17: 'qotd', 19: 'chargen',
        20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
        37: 'time', 42: 'name', 43: 'whois', 53: 'domain_u', 70: 'gopher',
        79: 'finger', 80: 'http', 109: 'pop_2', 110: 'pop_3', 111: 'sunrpc',
        113: 'auth', 119: 'nntp', 123: 'ntp_u', 135: 'loc_srv', 139: 'netbios_ssn',
        143: 'imap4', 179: 'bgp', 389: 'ldap', 443: 'https', 445: 'microsoft_ds',
        513: 'login', 514: 'shell', 515: 'printer', 540: 'uucp', 543: 'klogin',
        544: 'kshell', 993: 'imaps', 995: 'pop3s', 1080: 'socks', 1433: 'sql_net',
        1521: 'oracle', 1723: 'pptp', 2049: 'nfs', 3389: 'ms_term', 5432: 'postgresql',
        5900: 'vnc', 6000: 'X11', 8080: 'http_8080', 8443: 'https_alt'
    }
    
    udp_services = {
        7: 'echo', 9: 'discard', 13: 'daytime', 17: 'qotd', 19: 'chargen',
        37: 'time', 42: 'name', 53: 'domain', 67: 'bootps', 68: 'bootpc',
        69: 'tftp', 111: 'sunrpc', 123: 'ntp', 135: 'loc_srv', 137: 'netbios_ns',
        138: 'netbios_dgm', 161: 'snmp', 162: 'snmp_trap', 177: 'xdmcp',
        500: 'isakmp', 514: 'syslog', 520: 'route', 1900: 'upnp', 5353: 'mdns'
    }
    
    if protocol.lower() == 'tcp':
        return tcp_services.get(port, 'other')
    elif protocol.lower() == 'udp':
        return udp_services.get(port, 'other')
    else:
        return 'other'

def determine_flag(packet):
    """Determine connection flag based on TCP flags"""
    if not packet.haslayer(TCP):
        return "SF"  # Default for non-TCP
    
    tcp_flags = packet[TCP].flags
    
    # Parse TCP flags
    flags = {
        'FIN': bool(tcp_flags & 0x01),
        'SYN': bool(tcp_flags & 0x02),
        'RST': bool(tcp_flags & 0x04),
        'PSH': bool(tcp_flags & 0x08),
        'ACK': bool(tcp_flags & 0x10),
        'URG': bool(tcp_flags & 0x20)
    }
    
    # Determine flag based on combination
    if flags['RST']:
        return "RSTO" if flags['ACK'] else "RST"
    elif flags['SYN'] and not flags['ACK']:
        return "S0"  # SYN sent, no reply
    elif flags['SYN'] and flags['ACK']:
        return "S1"  # SYN-ACK sent
    elif flags['FIN']:
        return "SF"  # Normal termination
    else:
        return "SF"  # Assume successful

def extract_basic_features(packet):
    """Extract basic features from a single packet"""
    features = {
        'protocol_type': 'other',
        'service': 'other',
        'flag': 'SF',
        'src_bytes': 0,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0
    }
    
    if not packet.haslayer(IP):
        return features, None, None, None, None
    
    ip_layer = packet[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    
    # Check for land attack (same src and dst IP)
    features['land'] = 1 if src_ip == dst_ip else 0
    
    # Check for wrong fragments
    features['wrong_fragment'] = 1 if (ip_layer.flags & 0x2) and ip_layer.frag > 0 else 0
    
    # Protocol type
    if packet.haslayer(TCP):
        features['protocol_type'] = 'tcp'
        tcp_layer = packet[TCP]
        features['service'] = get_service_from_port(tcp_layer.dport, 'tcp')
        features['flag'] = determine_flag(packet)
        features['urgent'] = 1 if tcp_layer.flags & 0x20 else 0  # URG flag
        src_port = tcp_layer.sport
        dst_port = tcp_layer.dport
    elif packet.haslayer(UDP):
        features['protocol_type'] = 'udp'
        udp_layer = packet[UDP]
        features['service'] = get_service_from_port(udp_layer.dport, 'udp')
        features['flag'] = 'SF'
        src_port = udp_layer.sport
        dst_port = udp_layer.dport
    elif packet.haslayer(ICMP):
        features['protocol_type'] = 'icmp'
        features['service'] = 'eco_i'
        features['flag'] = 'SF'
        src_port = dst_port = 0
    else:
        src_port = dst_port = 0
    
    # Packet size
    packet_size = len(packet)
    features['src_bytes'] = packet_size
    features['dst_bytes'] = packet_size
    
    return features, src_ip, dst_ip, src_port, dst_port

def calculate_statistical_features(src_ip, dst_ip, src_port, dst_port, service, protocol):
    """Calculate statistical features based on connection history"""
    current_time = time.time()
    
    # Get connections in the last 2 seconds
    recent_connections = [
        conn for conn in connection_history 
        if current_time - conn['timestamp'] <= TIME_WINDOW
    ]
    
    # Count connections to same destination
    count = len([conn for conn in recent_connections if conn['dst_ip'] == dst_ip])
    
    # Count connections to same service
    srv_count = len([conn for conn in recent_connections 
                    if conn['service'] == service])
    
    # Calculate error rates
    serror_connections = [conn for conn in recent_connections 
                         if conn['flag'] in ['S0', 'S1', 'S2', 'S3']]
    serror_rate = len(serror_connections) / max(count, 1)
    
    srv_serror_connections = [conn for conn in recent_connections 
                             if conn['service'] == service and conn['flag'] in ['S0', 'S1', 'S2', 'S3']]
    srv_serror_rate = len(srv_serror_connections) / max(srv_count, 1)
    
    # Calculate rerror rates
    rerror_connections = [conn for conn in recent_connections 
                         if conn['flag'] in ['REJ', 'RSTO', 'RSTOS0', 'RSTR', 'SH']]
    rerror_rate = len(rerror_connections) / max(count, 1)
    
    srv_rerror_connections = [conn for conn in recent_connections 
                             if conn['service'] == service and conn['flag'] in ['REJ', 'RSTO', 'RSTOS0', 'RSTR', 'SH']]
    srv_rerror_rate = len(srv_rerror_connections) / max(srv_count, 1)
    
    # Calculate same service rate
    same_srv_rate = srv_count / max(count, 1)
    
    # Calculate different service rate
    diff_services = len(set(conn['service'] for conn in recent_connections))
    diff_srv_rate = (diff_services - 1) / max(count, 1) if count > 1 else 0
    
    # Calculate srv_diff_host_rate
    srv_diff_hosts = len(set(conn['dst_ip'] for conn in recent_connections 
                            if conn['service'] == service))
    srv_diff_host_rate = (srv_diff_hosts - 1) / max(srv_count, 1) if srv_count > 1 else 0
    
    return {
        'count': count,
        'srv_count': srv_count,
        'serror_rate': serror_rate,
        'srv_serror_rate': srv_serror_rate,
        'rerror_rate': rerror_rate,
        'srv_rerror_rate': srv_rerror_rate,
        'same_srv_rate': same_srv_rate,
        'diff_srv_rate': diff_srv_rate,
        'srv_diff_host_rate': srv_diff_host_rate
    }

def calculate_host_based_features(dst_ip, service, src_port):
    """Calculate host-based traffic features"""
    current_time = time.time()
    
    # Get recent connections to the same destination host
    recent_host_connections = [
        conn for conn in connection_history 
        if conn['dst_ip'] == dst_ip and current_time - conn['timestamp'] <= TIME_WINDOW
    ]
    
    dst_host_count = len(recent_host_connections)
    
    # Count connections to same service on same host
    dst_host_srv_count = len([conn for conn in recent_host_connections 
                             if conn['service'] == service])
    
    # Calculate same service rate for destination host
    dst_host_same_srv_rate = dst_host_srv_count / max(dst_host_count, 1)
    
    # Calculate different service rate for destination host
    dst_host_services = len(set(conn['service'] for conn in recent_host_connections))
    dst_host_diff_srv_rate = (dst_host_services - 1) / max(dst_host_count, 1) if dst_host_count > 1 else 0
    
    # Calculate same source port rate
    same_src_port_connections = len([conn for conn in recent_host_connections 
                                   if conn['src_port'] == src_port])
    dst_host_same_src_port_rate = same_src_port_connections / max(dst_host_count, 1)
    
    # Calculate service different host rate
    service_connections = [conn for conn in connection_history 
                          if conn['service'] == service and current_time - conn['timestamp'] <= TIME_WINDOW]
    service_hosts = len(set(conn['dst_ip'] for conn in service_connections))
    dst_host_srv_diff_host_rate = (service_hosts - 1) / max(len(service_connections), 1) if service_connections else 0
    
    # Calculate error rates for destination host
    dst_host_serror_connections = [conn for conn in recent_host_connections 
                                  if conn['flag'] in ['S0', 'S1', 'S2', 'S3']]
    dst_host_serror_rate = len(dst_host_serror_connections) / max(dst_host_count, 1)
    
    dst_host_srv_serror_connections = [conn for conn in recent_host_connections 
                                      if conn['service'] == service and conn['flag'] in ['S0', 'S1', 'S2', 'S3']]
    dst_host_srv_serror_rate = len(dst_host_srv_serror_connections) / max(dst_host_srv_count, 1)
    
    dst_host_rerror_connections = [conn for conn in recent_host_connections 
                                  if conn['flag'] in ['REJ', 'RSTO', 'RSTOS0', 'RSTR', 'SH']]
    dst_host_rerror_rate = len(dst_host_rerror_connections) / max(dst_host_count, 1)
    
    dst_host_srv_rerror_connections = [conn for conn in recent_host_connections 
                                      if conn['service'] == service and conn['flag'] in ['REJ', 'RSTO', 'RSTOS0', 'RSTR', 'SH']]
    dst_host_srv_rerror_rate = len(dst_host_srv_rerror_connections) / max(dst_host_srv_count, 1)
    
    return {
        'dst_host_count': dst_host_count,
        'dst_host_srv_count': dst_host_srv_count,
        'dst_host_same_srv_rate': dst_host_same_srv_rate,
        'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
        'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
        'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
        'dst_host_serror_rate': dst_host_serror_rate,
        'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
        'dst_host_rerror_rate': dst_host_rerror_rate,
        'dst_host_srv_rerror_rate': dst_host_srv_rerror_rate
    }

def extract_comprehensive_features(packet):
    """Extract all features from a packet with connection tracking"""
    # Extract basic features
    basic_features, src_ip, dst_ip, src_port, dst_port = extract_basic_features(packet)
    
    if not src_ip:  # Skip if no IP layer
        return None
    
    # Calculate duration (for now, set to 0 for single packets)
    duration = 0
    
    # Extract service and protocol
    service = basic_features['service']
    protocol = basic_features['protocol_type']
    
    # Calculate statistical features
    statistical_features = calculate_statistical_features(
        src_ip, dst_ip, src_port, dst_port, service, protocol
    )
    
    # Calculate host-based features
    host_features = calculate_host_based_features(dst_ip, service, src_port)
    
    # Record this connection
    connection_record = {
        'timestamp': time.time(),
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'service': service,
        'protocol': protocol,
        'flag': basic_features['flag']
    }
    connection_history.append(connection_record)
    
    # Combine all features
    features = {
        'duration': duration,
        'protocol_type': protocol,
        'service': service,
        'flag': basic_features['flag'],
        'src_bytes': basic_features['src_bytes'],
        'dst_bytes': basic_features['dst_bytes'],
        'land': basic_features['land'],
        'wrong_fragment': basic_features['wrong_fragment'],
        'urgent': basic_features['urgent'],
        'hot': 0,  # Requires content inspection
        'num_failed_logins': 0,  # Requires application layer analysis
        'logged_in': 0,  # Requires application layer analysis
        'num_compromised': 0,  # Requires application layer analysis
        'root_shell': 0,  # Requires application layer analysis
        'su_attempted': 0,  # Requires application layer analysis
        'num_root': 0,  # Requires application layer analysis
        'num_file_creations': 0,  # Requires application layer analysis
        'num_shells': 0,  # Requires application layer analysis
        'num_access_files': 0,  # Requires application layer analysis
        'num_outbound_cmds': 0,  # Requires application layer analysis
        'is_host_login': 0,  # Requires application layer analysis
        'is_guest_login': 0,  # Requires application layer analysis
        **statistical_features,
        **host_features
    }
    
    return features

def preprocess_features(features_dict):
    """Preprocess features for model prediction"""
    feature_columns = [col for col in ordered_features if col not in ["attack", "difficulty"]]
    data_row = {col: features_dict.get(col, 0) for col in feature_columns}
    df = pd.DataFrame([data_row], columns=feature_columns)

    try:
        # Load preprocessors
        with open('onehot_encoder.pkl', 'rb') as f:
            encoder = joblib.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = joblib.load(f)
        with open('selector.pkl', 'rb') as f:
            selector = joblib.load(f)
        
        # One-hot encode categorical features
        X_cat = encoder.transform(df[categorical_columns])
        
        # Numerical features
        X_num = df.drop(columns=categorical_columns).values
        
        # Combine features
        X_transformed = np.hstack([X_num, X_cat])
        
        # Scale features
        X_scaled = scaler.transform(X_transformed)
        
        # Select features
        X_selected = selector.transform(X_scaled)
        
        return X_selected
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def main():
    """Main detection loop"""
    print("Starting enhanced network intrusion detection system...")
    print("Monitoring network traffic for intrusions...")
    print("(Press CTRL+C to stop)")
    
    try:
        # Load model
        with open('Random_Forest_model.pkl', 'rb') as f:
            model = joblib.load(f)
        print("Model loaded successfully")
        
        # Check model classes to understand prediction labels
        if hasattr(model, 'classes_'):
            print(f"Model classes: {model.classes_}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    packet_count = 0
    normal_count = 0
    attack_count = 0
    
    try:
        while True:
            # Capture packets
            packets = sniff(count=5, timeout=5, iface="\\Device\\NPF_Loopback")

            
            if packets:
                for packet in packets:
                    packet_count += 1
                    
                    # Extract comprehensive features
                    features = extract_comprehensive_features(packet)
                    
                    if features:
                        # Preprocess features
                        processed_features = preprocess_features(features)
                        # 🔍 Debug print - See flags and counts
                        print(f"[#{packet_count}] Flag: {features['flag']} | Count: {features['count']} | Serror_rate: {features['serror_rate']:.2f}")

                    # 🚨 Rule-based fallback detection (SYN flood-like)
                        if features['flag'] == 'S0' and features['count'] > 10:
                            attack_count += 1
                            print(f"\n🚨 RULE-BASED ALERT: Possible SYN flood detected!")
                            print(f"Packet #{packet_count} | Flag: {features['flag']} | Count: {features['count']}")
                            print(f"Source IP: {packet[IP].src if packet.haslayer(IP) else 'Unknown'}")
                            print(f"Destination IP: {packet[IP].dst if packet.haslayer(IP) else 'Unknown'}")
                            print("-" * 50)
                            continue  # skip ML model for this packet

                        if processed_features is not None:
                            # Make prediction
                            prediction = model.predict(processed_features)
                            prediction_proba = model.predict_proba(processed_features)
                            print(f"🔮 Model Prediction: {prediction[0]} | Probabilities: {prediction_proba}")

                            # Determine if this is an attack or normal traffic
                            pred_value = prediction[0]
                            is_attack = False
                            
                            # Check different possible encodings for normal traffic
                            if hasattr(model, "predict_proba"):
                                is_attack = prediction_proba[0][1] > 0.5  # class 1 = attack
                            else:
                                is_attack = prediction[0] != 0  # fallback if no probabilities
                            
                            if is_attack:
                                attack_count += 1
                                print(f"\n🚨 ALERT: Potential attack detected!")
                                print(f"Packet #{packet_count} - Prediction: {prediction[0]}")
                                print(f"Confidence: {max(prediction_proba[0]):.2f}")
                                print(f"Source: {packet[IP].src if packet.haslayer(IP) else 'Unknown'}")
                                print(f"Destination: {packet[IP].dst if packet.haslayer(IP) else 'Unknown'}")
                                print(f"Service: {features['service']}")
                                print(f"Protocol: {features['protocol_type']}")
                                print(f"Connection count: {features['count']}")
                                print(f"Service count: {features['srv_count']}")
                                print(f"Error rate: {features['serror_rate']:.2f}")
                                print("-" * 50)
                            else:
                                normal_count += 1
                                if packet_count % 10 == 0:  # Print every 10th normal packet to reduce spam
                                    print(f"Packet #{packet_count}: Normal traffic - {features['service']} ({features['protocol_type']}) [Normal: {normal_count}, Attacks: {attack_count}]")
                    
                    # Clean old connections periodically
                    if packet_count % 100 == 0:
                        current_time = time.time()
                        cutoff_time = current_time - 60  # Keep connections from last minute
                        while connection_history and connection_history[0]['timestamp'] < cutoff_time:
                            connection_history.popleft()
                        
                        # Print summary
                        print(f"\n📊 Summary after {packet_count} packets:")
                        print(f"Normal traffic: {normal_count} ({normal_count/packet_count*100:.1f}%)")
                        print(f"Potential attacks: {attack_count} ({attack_count/packet_count*100:.1f}%)")
                        print("-" * 30)
            else:
                print("No packets captured in this cycle")
            
            time.sleep(1)  # Small delay between cycles
            
    except KeyboardInterrupt:
        print(f"\n\nFinal Summary:")
        print(f"Total packets analyzed: {packet_count}")
        print(f"Normal traffic: {normal_count} ({normal_count/packet_count*100:.1f}%)")
        print(f"Potential attacks: {attack_count} ({attack_count/packet_count*100:.1f}%)")
        print("\nStopping intrusion detection system...")
    except Exception as e:
        print(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()