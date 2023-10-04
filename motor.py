#!/usr/bin/env python3

import mraa
import time

ENABLE = 3 # GPIO pin number connected to the 
DIR_1 = 5 
DIR_2 = 7 

def main():
    enable = mraa.Gpio(ENABLE)
    enable.dir(mraa.DIR_OUT)
    dir1 = mraa.Gpio(DIR_1)
    dir1.dir(mraa.DIR_OUT)
    dir2 = mraa.Gpio(DIR_2)
    dir2.dir(mraa.DIR_OUT)
    adc = mraa.Aio(0)

    while True:
        print(adc.read())
        # Motor power off
        enable.write(False)
        di1.write(True)
        dir2.write(False)

        # Motor power on and going towards dir1
        enable.write(True)
        time.sleep(1)
        enable.write(False)

        #Change direction
        di1.write(False)
        dir2.write(True)

        # Motor power on and going towards dir2
        enable.write(True)
        time.sleep(1)


if __name__ == '__main__':
    main()