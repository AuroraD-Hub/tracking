#!/usr/bin/env python3

import mraa
import time

SERVO_PIN = 3 # GPIO pin number connected to the servo
MIN_PULSE = 544 # Minimum pulse width (in microseconds) for the servo
MAX_PULSE = 2400 # Maximum pulse width (in microseconds) for the servo
FREQUENCY = 50 # Servo motor frequency (in Hz)

def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def main():
    pwm = mraa.Pwm(SERVO_PIN)
    pwm.period_us(1000000 // FREQUENCY) # Set the PWM period in microseconds
    pwm.enable(True)
    while True:
        for angle in range(0, 180, 5):
            pulse = int(map_value(angle, 0, 180, MIN_PULSE, MAX_PULSE))
            pwm.pulsewidth_us(pulse)
            time.sleep(0.02)

        for angle in range(180, 0, -5):
            pulse = int(map_value(angle, 0, 180, MIN_PULSE, MAX_PULSE))
            pwm.pulsewidth_us(pulse)
            time.sleep(0.02)

if __name__ == '__main__':
    main()