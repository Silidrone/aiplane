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
        self.add_debug_message("=== EASYPYTHON STARTING ===")
        self.create_display_window()
        self.add_debug_message("EasyPython plugin started!")

    def after_physics(self):
        try:
            current_time = time.time()
            
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
        self.add_debug_message(f"--- Debug cleared at {current_time:.0f} ---")
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
            
            xp.drawString(color, left + 5, y_pos, "EasyPython - LTBJ Landing RL", 0, xp.Font_Basic)
            y_pos -= 20
            
            xp.drawString(color, left + 5, y_pos, f"Episodes: {self.episode_count}  --  CLICK TO READ STATE", 0, xp.Font_Basic)
            y_pos -= 25
            
            # Debug messages
            for message in self.debug_messages[-8:]:
                xp.drawString(color, left + 5, y_pos, message[:65], 0, xp.Font_Basic)
                y_pos -= 12
                
        except Exception as e:
            pass
        
    def mouse_click_callback(self, inWindowID, x, y, inMouse, inRefcon):
        if inMouse == xp.MouseDown:
            self.add_debug_message("*** MOUSE CLICKED! Reading state... ***")
            self.read_current_state()
            self.reset_to_approach()
        return 1
        
    def key_callback(self, inWindowID, inKey, inFlags, inVirtualKey, inRefcon, losingFocus):
        pass
        
    def cursor_callback(self, inWindowID, x, y, inRefcon):
        return xp.CursorDefault

    def read_current_state(self):
        try:
            self.add_debug_message("Reading current aircraft state...")
            
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
            
            xplane_path = xp.getSystemPath()
            file_path = xplane_path + "abc.txt"
            with open(file_path, "w") as f:
                f.write("Aircraft State Data\n")
                f.write("==================\n\n")
                f.write(f"Quaternion: q0={quaternion[0]:.6f}, q1={quaternion[1]:.6f}, q2={quaternion[2]:.6f}, q3={quaternion[3]:.6f}\n")
                f.write(f"Local Position: x={local_x:.6f}, y={local_y:.6f}, z={local_z:.6f}\n")
                f.write(f"Local Velocity: vx={local_vx:.6f}, vy={local_vy:.6f}, vz={local_vz:.6f}\n")
                f.write(f"Angular Velocity: P={P:.6f}, Q={Q:.6f}, R={R:.6f}\n")
                f.write(f"Lat/Lon/Elevation: {lat:.6f}, {lon:.6f}, {elevation:.6f}\n")
                
            self.add_debug_message("State saved to abc.txt!")
            
        except Exception as e:
            self.add_debug_message(f"READ ERROR: {e}")

    def reset_to_approach(self):
        try:
            self.add_debug_message("RESET: Using hardcoded 3nm approach state...")
            
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
            self.add_debug_message(f"RESET: Complete - Episode #{self.episode_count}!")
            
        except Exception as e:
            self.add_debug_message(f"RESET ERROR: {e}")