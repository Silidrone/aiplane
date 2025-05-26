from XPPython3 import xp
from XPPython3.utils.easy_python import EasyPython
import math
import time

class PythonInterface(EasyPython):
    def __init__(self):
        super().__init__()
        self.name = "Airplane RL Landing Plugin"
        self.description = "Reinforcement Learning environment for landing training"
        self.id = "com.example.airplane_rl"
        self.airport_icao = "LTBJ"
        
        self.window_id = None
        self.episode_count = 0
        self.status_message = "Starting..."
        
        # Debug messages
        self.debug_messages = []
        self.max_debug_messages = 8
        self.last_clear_time = time.time()

    def onStart(self):
        # Cache DataRef handles
        self.q_ref = xp.findDataRef("sim/flightmodel/position/q")
        self.local_x_ref = xp.findDataRef("sim/flightmodel/position/local_x")
        self.local_y_ref = xp.findDataRef("sim/flightmodel/position/local_y")
        self.local_z_ref = xp.findDataRef("sim/flightmodel/position/local_z")
        self.local_vx_ref = xp.findDataRef("sim/flightmodel/position/local_vx")
        self.local_vy_ref = xp.findDataRef("sim/flightmodel/position/local_vy")
        self.local_vz_ref = xp.findDataRef("sim/flightmodel/position/local_vz")
        self.P_ref = xp.findDataRef("sim/flightmodel/position/P")
        self.Q_ref = xp.findDataRef("sim/flightmodel/position/Q")
        self.R_ref = xp.findDataRef("sim/flightmodel/position/R")
        self.lat_ref = xp.findDataRef("sim/flightmodel/position/latitude")
        self.lon_ref = xp.findDataRef("sim/flightmodel/position/longitude")
        self.elevation_ref = xp.findDataRef("sim/flightmodel/position/elevation")
        self.create_display_window()

    def after_physics(self):
        try:
            current_time = time.time()
            
            # Print current state every frame
            self.read_current_state()
            
            # Clear debug messages every 10 seconds
            if current_time - self.last_clear_time >= 10.0:
                self.clear_debug_messages()
                    
        except Exception as e:
            self.add_debug_message(f"Error: {e}")

    def add_debug_message(self, message):
        self.debug_messages.append(message)
        if len(self.debug_messages) > self.max_debug_messages:
            self.debug_messages.pop(0)

    def clear_debug_messages(self):
        self.debug_messages = []
        current_time = time.time()
        self.last_clear_time = current_time

    def create_display_window(self):
        try:
            self.window_id = xp.createWindowEx(50, 700, 600, 100, 1,
                                             self.draw_window_callback,
                                             self.mouse_click_callback,
                                             self.key_callback,
                                             self.cursor_callback,
                                             None, 0,
                                             xp.WindowDecorationRoundRectangle,
                                             xp.WindowLayerFloatingWindows,
                                             None)
        except Exception as e:
            self.add_debug_message(f"ERROR creating window: {e}")
        
    def draw_window_callback(self, inWindowID, inRefcon):
        try:
            left, top, right, bottom = xp.getWindowGeometry(inWindowID)
            xp.drawTranslucentDarkBox(left, top, right, bottom)
            
            color = (1.0, 1.0, 1.0)
            y_pos = top - 15
            
            xp.drawString(color, left + 5, y_pos, f"Episodes: {self.episode_count}", 0, xp.Font_Basic)
            y_pos -= 20
            
            # Debug messages
            for message in self.debug_messages[-8:]:
                xp.drawString(color, left + 5, y_pos, message[:65], 0, xp.Font_Basic)
                y_pos -= 12
                
        except Exception as e:
            pass
        
    def mouse_click_callback(self, inWindowID, x, y, inMouse, inRefcon):
        if inMouse == xp.MouseDown:
            self.reset_to_approach()
        return 1
        
    def key_callback(self, inWindowID, inKey, inFlags, inVirtualKey, inRefcon, losingFocus):
        pass
        
    def cursor_callback(self, inWindowID, x, y, inRefcon):
        return xp.CursorDefault

    def read_current_state(self):
        try:
            quaternion = []
            xp.getDatavf(self.q_ref, quaternion, 0, 4)
            local_x = xp.getDataf(self.local_x_ref)
            local_y = xp.getDataf(self.local_y_ref)
            local_z = xp.getDataf(self.local_z_ref)
            local_vx = xp.getDataf(self.local_vx_ref)
            local_vy = xp.getDataf(self.local_vy_ref)
            local_vz = xp.getDataf(self.local_vz_ref)
            P = xp.getDataf(self.P_ref)
            Q = xp.getDataf(self.Q_ref)
            R = xp.getDataf(self.R_ref)
            lat = xp.getDataf(self.lat_ref)
            lon = xp.getDataf(self.lon_ref)
            elevation = xp.getDataf(self.elevation_ref)
            self.add_debug_message(f"Q: {quaternion[0]:.3f},{quaternion[1]:.3f},{quaternion[2]:.3f},{quaternion[3]:.3f}")
            self.add_debug_message(f"Pos: x={local_x:.1f}, y={local_y:.1f}, z={local_z:.1f}")
            self.add_debug_message(f"Vel: vx={local_vx:.1f}, vy={local_vy:.1f}, vz={local_vz:.1f}")
            self.add_debug_message(f"AngVel: P={P:.3f}, Q={Q:.3f}, R={R:.3f}")
            self.add_debug_message(f"Lat/Lon/Elev: {lat:.6f}, {lon:.6f}, {elevation:.1f}")
        except Exception as e:
            self.add_debug_message(f"READ ERROR: {e}")

    def reset_to_approach(self):
        try:
            # Hardcoded values from captured 3nm approach state
            xp.setDatavf(self.q_ref, [0.993124, -0.000318, 0.005706, -0.116929], 0, 4)
            xp.setDataf(self.local_x_ref, 15389.733398)
            xp.setDataf(self.local_y_ref, 329.115723)
            xp.setDataf(self.local_z_ref, 29976.988281)
            xp.setDataf(self.local_vx_ref, -7.377071)
            xp.setDataf(self.local_vy_ref, -1.748070)
            xp.setDataf(self.local_vz_ref, -31.024368)
            xp.setDataf(self.P_ref, -0.650009)
            xp.setDataf(self.Q_ref, 1.183697)
            xp.setDataf(self.R_ref, -0.314619)
            self.episode_count += 1
        except Exception as e:
            self.add_debug_message(f"RESET ERROR: {e}")