#!/usr/bin/env python3
"""ROS 2 bridge: /cmd_vel -> ESP32 over USB serial; ESP32 telemetry -> /odom, TF, /cliff.
Run: python3 esp32_bridge.py --ros-args -p port:=/dev/ttyUSB0
Needs: pip install pyserial; ros-<distro>-tf2-ros
"""
import math
import time

import rclpy
import serial
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class Esp32Bridge(Node):
    def __init__(self):
        super().__init__('esp32_bridge')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('ticks_per_rev', 660.0)   # must match firmware
        self.declare_parameter('wheel_radius', 0.0425)
        port = self.get_parameter('port').value
        self.tpr = self.get_parameter('ticks_per_rev').value
        self.r = self.get_parameter('wheel_radius').value

        self.ser = serial.Serial(port, 115200, timeout=0.02)
        time.sleep(2.0)                      # ESP32 resets when the port opens
        self.ser.reset_input_buffer()

        self.lin = 0.0
        self.ang = 0.0
        self.prev = None                     # (encL, encR, time)
        self.x = self.y = self.th = 0.0

        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.cliff_pub = self.create_publisher(Bool, 'cliff', 10)
        self.tf = TransformBroadcaster(self)
        self.create_timer(0.05, self.send_cmd)    # 20 Hz keeps ESP32 watchdog fed
        self.create_timer(0.01, self.read_serial)

    def on_cmd(self, msg):
        self.lin = max(-0.5, min(0.5, msg.linear.x))
        self.ang = max(-2.0, min(2.0, msg.angular.z))

    def send_cmd(self):
        self.ser.write(f'V,{self.lin:.3f},{self.ang:.3f}\n'.encode())

    def read_serial(self):
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors='ignore').strip()
            p = line.split(',')
            if len(p) != 6 or p[0] != 'T':
                continue
            try:
                el, er = int(p[1]), int(p[2])
                yaw = math.radians(float(p[3]))
                cliff = bool(int(p[4])) or bool(int(p[5]))
            except ValueError:
                continue
            self.update_odom(el, er, yaw)
            self.cliff_pub.publish(Bool(data=cliff))

    def update_odom(self, el, er, yaw):
        now = time.monotonic()
        if self.prev is None:
            self.prev = (el, er, now)
            self.th = yaw
            return
        pl, pr, pt = self.prev
        dt = max(now - pt, 1e-3)
        k = 2.0 * math.pi * self.r / self.tpr
        ds = ((el - pl) + (er - pr)) / 2.0 * k
        dth = yaw - self.th
        self.th = yaw
        self.x += ds * math.cos(self.th)
        self.y += ds * math.sin(self.th)
        self.prev = (el, er, now)

        stamp = self.get_clock().now().to_msg()
        qz, qw = math.sin(self.th / 2.0), math.cos(self.th / 2.0)

        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = 'odom'
        o.child_frame_id = 'base_link'
        o.pose.pose.position.x = self.x
        o.pose.pose.position.y = self.y
        o.pose.pose.orientation.z = qz
        o.pose.pose.orientation.w = qw
        o.twist.twist.linear.x = ds / dt
        o.twist.twist.angular.z = dth / dt
        self.odom_pub.publish(o)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf.sendTransform(t)

    def destroy_node(self):
        try:
            self.ser.write(b'S\n')
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = Esp32Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
