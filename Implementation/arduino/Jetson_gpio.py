import atexit
import random
import threading
import time

import RPi.GPIO as GPIO

# Used Pins
PUMP_PIN = 40
SERVO_ENABLE_PIN = 38
SOUND_ENABLE_PIN = 37
LIGHT_ENABLE_PIN = 36

# Testing
TEST_PIN = 12

GPIO.setmode(GPIO.BOARD)

class ToggleThread(threading.Thread):
    def __init__(self, pin, sleep_time=0, random_range=(0, 0)):
        threading.Thread.__init__(self)
        self.pin = pin
        GPIO.setup(self.pin, GPIO.OUT)
        self.sleep_time = sleep_time
        self.random_range = random_range
        self.running = False
        self.daemon = True
        self.start()

    def run(self):
        while True:
            if self.running:
                if self.sleep_time > 0:
                    print(f"Toggling to not {GPIO.input(self.pin)}")
                    GPIO.output(self.pin, not GPIO.input(self.pin))
                    time.sleep(self.sleep_time + random.uniform(self.random_range[0], self.random_range[1]))
                else:
                    GPIO.output(self.pin, True)

    def stop(self):
        print("Toggling to stop")
        self.running = False
        GPIO.output(self.pin, False)

    def start_toggle(self):
        print(f"Toggling to continue")
        self.running = True

# bei 250Hz -> 100/20/ neutral: 60 Laut GPT Auflösung 12 Bit -> max 3276 einstellbare schritte -> 0,055 grad Einstellbar -> Servo macht 0,09
class PWMThread():
    def __init__(self, pin, frequency=250, duty_cycle=0):
        self.pin = pin
        GPIO.setup(self.pin, GPIO.OUT)
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        self.pwm = GPIO.PWM(self.pin, self.frequency)

    def stop(self):
        print("PWM output stopping")
        self.pwm.stop()

    def set_frequency(self, frequency):
        if frequency <40 or frequency > 1000:
            print("PWM output frequency must be between 40 and 1000 Hz")
            return
        self.frequency = frequency
        self.pwm.ChangeFrequency(self.frequency)

    def set_duty_cycle(self, duty_cycle):
        self.duty_cycle = duty_cycle
        self.pwm.ChangeDutyCycle(self.duty_cycle)

    def start_pwm(self):
        print("PWM output starting")
        self.pwm.start(self.duty_cycle)

PUMP = ToggleThread(PUMP_PIN)
SERVO_ENABLE = ToggleThread(SERVO_ENABLE_PIN)
SOUND_ENABLE = ToggleThread(SOUND_ENABLE_PIN)
LIGHT_ENABLE = ToggleThread(LIGHT_ENABLE_PIN, 0.1, (0, 0,5))

threadi_one = ToggleThread(TEST_PIN, 1, (-0.5, 0.5))

HORIZONTAL_SERVO = PWMThread(33)
VERTICAL_SERVO = PWMThread(32)

def cleanup():
    print("Cleanup GPIOS")
    threadi_one.stop()
    PUMP.stop()
    SERVO_ENABLE.stop()
    SOUND_ENABLE.stop()
    LIGHT_ENABLE.stop()
    GPIO.cleanup()

# Register the cleanup function with atexit
atexit.register(cleanup)

if __name__ == "__main__":
    threadi_one.start_toggle()

    time.sleep(5)

    threadi_one.stop()
    time.sleep(5)
    threadi_one.start_toggle()

    time.sleep(5)


