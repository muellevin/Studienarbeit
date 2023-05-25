import serial
import atexit
import threading
from time import sleep

class SerialCom(threading.Thread):
    
    vertical_pos = 0.0
    horizontal_pos = 0.0
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)
        self.running = False
        self.daemon = True
        self.start()
    
    def run(self):
        while True:
            if self.running:
                angles = f'v:{self.vertical_pos:.2f} h:{self.horizontal_pos:.2f}'
                self.ser.write(str(angles).encode('utf-8'))
                sleep(0.01)
                
    def stop(self):
        print("Toggling to stop")
        self.running = False
        self.ser.write(str("g").encode('utf-8'))

    def start_toggle(self):
        print(f"Toggling to continue")
        self.running = True

SERIAL_COM = SerialCom()

def cleanup():
    print("Cleanup Serial")
    SERIAL_COM.stop()
    SERIAL_COM.ser.close()

# Register the cleanup function with atexit
atexit.register(cleanup)

if __name__ == "__main__":
    SERIAL_COM.start_toggle()
    
    sleep(5)
    
    SERIAL_COM.stop()
