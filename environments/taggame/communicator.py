import json
import signal
import socket
from typing import Optional


class Communicator:
    RESET = "reset"
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'Communicator':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if Communicator._instance is not None:
            raise RuntimeError("Use get_instance() instead of constructor")
        
        self.sock: Optional[socket.socket] = None
        
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    
    def connect_to_server(self, host: str, port: int) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))
            print(f"Connected to server at {host}:{port}")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"Connection to server failed: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None
            return False
    
    def disconnect(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Disconnected from server.")
    
    def receive_state(self) -> str:
        if not self.sock:
            raise RuntimeError("Not connected to server")
        
        try:
            buffer = bytearray(4096)
            bytes_received = self.sock.recv_into(buffer)
            
            if bytes_received > 0:
                return buffer[:bytes_received].decode('utf-8')
            elif bytes_received == 0:
                raise RuntimeError("Server closed the connection.")
            else:
                raise RuntimeError("Error receiving state.")
        except (socket.error, socket.timeout) as e:
            raise RuntimeError(f"Socket error: {e}")
    
    def send_action(self, action: str) -> None:
        if not self.sock:
            raise RuntimeError("Not connected to server")
        
        try:
            action_with_newline = action + "\n"
            bytes_sent = self.sock.send(action_with_newline.encode('utf-8'))
            
            if bytes_sent <= 0:
                raise RuntimeError("Error sending action.")
        except (socket.error, socket.timeout) as e:
            raise RuntimeError(f"Socket error: {e}")
    
    def __del__(self) -> None:
        self.disconnect()