"""
Tricking Colab for activity tracking
"""

from pynput.mouse import Controller, Button
from datetime import datetime
import time

mouse = Controller()

# clicks every 2 minutes once
while True:
    mouse.click(Button.left, 1)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time, ': mouse clicked')

    time.sleep(120)