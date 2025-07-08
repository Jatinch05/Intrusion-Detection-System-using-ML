# Intrusion Detection System using Machine Learning

## Overview
An end-to-end Intrusion Detection System (IDS) that leverages machine learning to monitor network traffic, extract comprehensive features, and detect potential attacks in real-time.

## Features
- **Real-Time Packet Capture**: Uses Scapy to sniff live network traffic.
- **Comprehensive Feature Extraction**: Derives 40+ statistical and host-based features per packet.
- **Preprocessing Pipeline**: Applies one-hot encoding, scaling, and feature selection.
- **Machine Learning Models**: Supports Random Forest, Gradient Boosting, Extra Trees, AdaBoost, and MLP.
- **Live Inference**: Loads trained models for on-the-fly predictions and alerts.
- **Summary Reporting**: Periodic statistics of normal vs. attack packets.

## Repository Structure
```
.
├── KDDTrain+.txt               # Training dataset (NSL-KDD)
├── KDDTest+.txt                # Test dataset (NSL-KDD)
├── main.ipynb                  # Notebook: preprocessing, training, evaluation
├── onehot_encoder.pkl          # Categorical encoder
├── scaler.pkl                  # Feature scaler
├── selector.pkl                # Feature selector
├── Random_Forest_model.pkl     # Trained Random Forest model
├── Gradient_Boost_model.pkl    # Trained Gradient Boosting model
├── Extra_Trees_model.pkl       # Trained Extra Trees model
├── AdaBoost_model.pkl          # Trained AdaBoost model
├── MLP_model.pkl               # Trained MLP model
├── attack.py                   # Script to simulate network attacks
├── demo.py                     # Utility: list local network interfaces
├── test.py                     # Real-time IDS execution and alerting
└── README.md                   # Project documentation
```

## Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/Jatinch05/Intrusion-Detection-System-using-ML.git
   cd Intrusion-Detection-System-using-ML
   ```

2. **Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preprocessing & Model Training
Open and run `main.ipynb` to:
- Load and preprocess NSL-KDD data
- Train and evaluate candidate ML models
- Save the best-performing model and preprocessors

### 2. Simulate Attacks
Use `attack.py` to generate synthetic network attacks:
```bash
python3 attack.py <target_ip> --attack-type syn --count 200
```

### 3. Real-Time Detection
Run the main IDS loop via `test.py`:
```bash
python3 test.py
```
- Monitors live traffic on the default interface.
- Prints debug info and ML-based alerts.
- Summarizes normal vs. attack statistics every 100 packets.

## Customization
- **Interface**: Modify `iface` parameter in `sniff()` call.
- **Model**: Swap `Random_Forest_model.pkl` with any supported model.


## Future Work
- Add application-layer feature extraction (login attempts, file accesses).
- Integrate a dashboard for real-time monitoring and visualization.
- Implement unit tests and CI/CD pipeline.



## License
MIT License
