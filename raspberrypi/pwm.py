'''
******************************************************************

    Project Description: PWM Waveform Output for Gimbal Control

    Hardware information:
    ---------------------------------------------------------
        Gimbal               |              3D print
        Gimbal Servos        |              ST90S*2 (2-DOF)
        Servos Frequency     |              50Hz
        Servos Angle Range   |              180 degree
        Servos Pulse Range   |              500us ~ 2500us
    ---------------------------------------------------------

    OutPins information (Can be modified as needed):
    ---------------------------------------------------------
        Pin_x                |              Pin17
        Pin_y                |              Pin18
    ---------------------------------------------------------

    Author: Veinsure Lee
    Edit Date: 2025.03.28

******************************************************************
'''

import RPi.GPIO as GPIO
import time

# set GPIO model and PWM pram
GPIO.setmode(GPIO.BCM)
pwm_pin_x = 17
pwm_pin_y = 18
frequency = 50

Duty_Max = 2500 * 50 * 1e-6 * 100  # 12.5%
Duty_Min = 500 * 50 * 1e-6 * 100  # 2.5%
Duty_Step = (Duty_Max - Duty_Min) / 100  # 0.1%step

# initialize control parm
Cnt_x = 0
Cnt_StepNum_x = 100  # step total
Wait_Time_x = 0.1  # step

# set GPIO and PWM channel
GPIO.setup(pwm_pin_x, GPIO.OUT)
GPIO.setup(pwm_pin_y, GPIO.OUT)
pwm_x = GPIO.PWM(pwm_pin_x, frequency)
pwm_y = GPIO.PWM(pwm_pin_y, frequency)


def main():
    global Cnt_x

    try:
        pwm_x.start(Duty_Min)
        while True:
            duty = Duty_Min + Cnt_x * Duty_Step
            pwm_x.ChangeDutyCycle(duty)

            Cnt_x = 0 if Cnt_x >= Cnt_StepNum_x else Cnt_x + 1

            print(f"Position: {Cnt_x}, Duty: {duty:.1f}%")

            time.sleep(Wait_Time_x)

    except KeyboardInterrupt:

        pwm_x.stop()
        pwm_y.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()