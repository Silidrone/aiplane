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
            
            xp.drawString(color, left + 5, y_pos, f"Episodes: {self.episode_count}  --  CLICK TO RESET", 0, xp.Font_Basic)
            y_pos -= 25
            
            # Debug messages
            for message in self.debug_messages[-8:]:
                xp.drawString(color, left + 5, y_pos, message[:65], 0, xp.Font_Basic)
                y_pos -= 12
                
        except Exception as e:
            pass
        
    def mouse_click_callback(self, inWindowID, x, y, inMouse, inRefcon):
        if inMouse == xp.MouseDown:
            self.add_debug_message("*** MOUSE CLICKED! Starting reset... ***")
            self.reset_to_approach()
        return 1
        
    def key_callback(self, inWindowID, inKey, inFlags, inVirtualKey, inRefcon, losingFocus):
        pass
        
    def cursor_callback(self, inWindowID, x, y, inRefcon):
        return xp.CursorDefault

    def reset_to_approach(self):
        try:
            self.add_debug_message("RESET: Direct quaternion method...")
            
            heading_deg = 160.0
            pitch_deg = -3.0
            roll_deg = 0.0
            
            heading_rad = math.radians(heading_deg)
            pitch_rad = math.radians(pitch_deg)
            roll_rad = math.radians(roll_deg)
            
            psi_half = heading_rad / 2
            theta_half = pitch_rad / 2  
            phi_half = roll_rad / 2
            
            q0 = math.cos(psi_half) * math.cos(theta_half) * math.cos(phi_half) + math.sin(psi_half) * math.sin(theta_half) * math.sin(phi_half)
            q1 = math.cos(psi_half) * math.cos(theta_half) * math.sin(phi_half) - math.sin(psi_half) * math.sin(theta_half) * math.cos(phi_half)
            q2 = math.cos(psi_half) * math.sin(theta_half) * math.cos(phi_half) + math.sin(psi_half) * math.cos(theta_half) * math.sin(phi_half)
            q3 = -math.cos(psi_half) * math.sin(theta_half) * math.sin(phi_half) + math.sin(psi_half) * math.cos(theta_half) * math.cos(phi_half)
            
            q_ref = xp.findDataRef("sim/flightmodel/position/q")
            xp.setDatavf(q_ref, [q0, q1, q2, q3], 0, 4)
            
            target_lat = 38.3347
            target_lon = 27.1583 
            target_alt = 1500
            local_x, local_y, local_z = xp.worldToLocal(target_lat, target_lon, target_alt)
            
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_x"), local_x)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_y"), local_y)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_z"), local_z)
            
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_vx"), 0)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_vy"), 0)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/local_vz"), 0)
            
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/P"), 0.0)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/Q"), 0.0)
            xp.setDataf(xp.findDataRef("sim/flightmodel/position/R"), 0.0)
            
            self.episode_count += 1
            self.add_debug_message(f"RESET: Complete - Episode #{self.episode_count}!")
            
        except Exception as e:
            self.add_debug_message(f"RESET ERROR: {e}")