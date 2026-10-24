#!/usr/bin/env python3
"""Pi-side motor control for the ESP32 slave (no ROS needed).

  python3 motor_control.py teleop     keyboard driving  (w/s fwd/back, a/d turn, space stop, q quit)
  python3 motor_control.py square     drive a 0.5 m square (robot needs ~1.5 m clear space!)
  python3 motor_control.py            prints live telemetry

In your own code:
    from motor_control import Motors
    m = Motors('/dev/ttyUSB0')
    m.drive_distance(0.5)       # metres, straight
    m.turn_angle(90)            # degrees, + = left (counter-clockwise)
    m.drive(0.2, 0.0)           # continuous velocity until you call m.stop()
"""
import math
import sys
import threading
import time

import serial

TICKS_PER_REV = 660.0        # must match the firmware
WHEEL_RADIUS = 0.0425        # m


class Motors:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(2.0)                       # ESP32 resets when the port opens
        self.ser.reset_input_buffer()
        self.lin = 0.0
        self.ang = 0.0
        self.tel = dict(encL=0, encR=0, yaw=0.0, cliffL=0, cliffR=0,
                        tension=0.0, batt=0.0, flags=0)
        self._run = True
        threading.Thread(target=self._tx, daemon=True).start()
        threading.Thread(target=self._rx, daemon=True).start()

    # ---- low level ----
    def _tx(self):                            # resend 20x/s so the ESP32 watchdog stays happy
        while self._run:
            self.ser.write(f'V,{self.lin:.3f},{self.ang:.3f}\n'.encode())
            time.sleep(0.05)

    def _rx(self):
        while self._run:
            try:
                p = self.ser.readline().decode(errors='ignore').strip().split(',')
                if p[0] == 'T' and len(p) == 6:
                    self.tel.update(encL=int(p[1]), encR=int(p[2]), yaw=float(p[3]),
                                    cliffL=int(p[4]), cliffR=int(p[5]))
                elif p[0] == 'E' and len(p) == 4:
                    self.tel.update(tension=float(p[1]), batt=float(p[2]), flags=int(p[3]))
            except (ValueError, serial.SerialException):
                pass

    def send(self, line):                     # raw command, e.g. 'P,120,250' or 'H,1'
        self.ser.write((line + '\n').encode())

    # ---- basic motion ----
    def drive(self, lin, ang=0.0):
        self.lin = max(-0.5, min(0.5, lin))
        self.ang = max(-2.0, min(2.0, ang))

    def stop(self):
        self.lin = self.ang = 0.0
        self.send('S')

    def close(self):
        self._run = False
        self.stop()
        time.sleep(0.1)
        self.ser.close()

    @property
    def cliff(self):
        return bool(self.tel['cliffL'] or self.tel['cliffR'])

    # ---- closed-loop moves ----
    def drive_for(self, lin, ang, seconds):
        self.drive(lin, ang)
        time.sleep(seconds)
        self.stop()

    def drive_distance(self, meters, speed=0.2, timeout=30.0):
        """Straight move using encoder feedback. Negative meters = reverse. Aborts on cliff."""
        k = 2 * math.pi * WHEEL_RADIUS / TICKS_PER_REV
        start = (self.tel['encL'] + self.tel['encR']) / 2.0
        direction = 1 if meters >= 0 else -1
        t0 = time.time()
        while time.time() - t0 < timeout:
            done = ((self.tel['encL'] + self.tel['encR']) / 2.0 - start) * k * direction
            remaining = abs(meters) - done
            if remaining <= 0.005 or (direction > 0 and self.cliff):
                break
            v = min(speed, max(0.08, remaining * 1.5))        # slow down near the target
            self.drive(direction * v, 0.0)
            time.sleep(0.02)
        self.stop()
        return not self.cliff

    def turn_angle(self, degrees, speed=0.8, timeout=15.0):
        """In-place turn using gyro yaw. + = left (counter-clockwise)."""
        target = self.tel['yaw'] + degrees
        t0 = time.time()
        while time.time() - t0 < timeout:
            err = target - self.tel['yaw']
            if abs(err) < 1.5:
                break
            w = math.copysign(min(speed, max(0.3, abs(err) / 45.0)), err)
            self.drive(0.0, w)
            time.sleep(0.02)
        self.stop()


def teleop(m):
    import curses

    def run(scr):
        scr.nodelay(True)
        scr.addstr(0, 0, 'w/s forward/back  a/d turn  space stop  q quit')
        while True:
            c = scr.getch()
            if c == ord('w'): m.drive(0.2, 0.0)
            elif c == ord('s'): m.drive(-0.2, 0.0)
            elif c == ord('a'): m.drive(0.0, 0.8)
            elif c == ord('d'): m.drive(0.0, -0.8)
            elif c == ord(' '): m.stop()
            elif c == ord('q'): break
            t = m.tel
            scr.addstr(2, 0, f"cmd {m.lin:+.2f} m/s {m.ang:+.2f} rad/s   yaw {t['yaw']:7.1f}   "
                             f"cliff {t['cliffL']}{t['cliffR']}   batt {t['batt']:.1f}V   ")
            time.sleep(0.05)
    curses.wrapper(run)


if __name__ == '__main__':
    m = Motors(sys.argv[2] if len(sys.argv) > 2 else '/dev/ttyUSB0')
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else 'status'
        if mode == 'teleop':
            teleop(m)
        elif mode == 'square':
            for _ in range(4):
                m.drive_distance(0.5)
                m.turn_angle(90)
        else:
            while True:
                print(m.tel)
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        m.close()
