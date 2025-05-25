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
        
        self.window_id = None
        self.episode_count = 0
        self.status_message = "Starting..."
        
        # Debug messages
        self.debug_messages = []
        self.max_debug_messages = 8
        self.last_clear_time = time.time()

    def onStart(self):
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
            q_ref = xp.findDataRef("sim/flightmodel/position/q")
            quaternion = []
            xp.getDatavf(q_ref, quaternion, 0, 4)
            
            local_x = xp.getDataf(xp.findDataRef("sim/flightmodel/position/local_x"))
            local_y = xp.getDataf(xp.findDataRef("sim/flightmodel/position/local_y"))
            local_z = xp.getDataf(xp.findDataRef("sim/flightmodel/position/local_z"))
            
            local_vx = xp.getDataf(xp.findDataRef("sim/flightmodel/position/local_vx"))
            local_vy = xp.getDataf(xp.findDataRef("sim/flightmodel/position/local_vy"))
            local_vz = xp.getDataf(xp.findDataRef("sim/flightmodel/position/local_vz"))
            
            P = xp.getDataf(xp.findDataRef("sim/flightmodel/position/P"))
            Q = xp.getDataf(xp.findDataRef("sim/flightmodel/position/Q"))
            R = xp.getDataf(xp.findDataRef("sim/flightmodel/position/R"))
            
            lat = xp.getDataf(xp.findDataRef("sim/flightmodel/position/latitude"))
            lon = xp.getDataf(xp.findDataRef("sim/flightmodel/position/longitude"))
            elevation = xp.getDataf(xp.findDataRef("sim/flightmodel/position/elevation"))
            
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
            q_ref = xp.findDataRef("sim/flightmodel/position/q")
            xp.setDatavf(q_ref, [0.993124, -0.000318, 0.005706, -0.116929], 0, 4)
            
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_x"), 15389.733398)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_y"), 329.115723)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_z"), 29976.988281)
            
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_vx"), -7.377071)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_vy"), -1.748070)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_vz"), -31.024368)
            
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/P"), -0.650009)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/Q"), 1.183697)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/R"), -0.314619)
            
            self.episode_count += 1
        except Exception as e:
            self.add_debug_message(f"RESET ERROR: {e}")