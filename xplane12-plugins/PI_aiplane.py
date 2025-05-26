from XPPython3 import xp
from XPPython3.utils.easy_python import EasyPython
import time
from math import radians, sin, cos, sqrt, atan2

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
        self.max_debug_messages = 16
        self.last_clear_time = time.time()

        # LTBJ RWY 16
        self.RUNWAY_LAT = 38.2784
        self.RUNWAY_LON = 27.1612
        self.RUNWAY_HEADING = 160.0  # degrees

        # Cache all DataRefs used in state/action
        self.lat_ref = xp.findDataRef("sim/flightmodel/position/latitude")
        self.lon_ref = xp.findDataRef("sim/flightmodel/position/longitude")
        self.elevation_ref = xp.findDataRef("sim/flightmodel/position/elevation")
        self.pitch_ref = xp.findDataRef("sim/flightmodel/position/theta")
        self.bank_ref = xp.findDataRef("sim/flightmodel/position/phi")
        self.heading_ref = xp.findDataRef("sim/flightmodel/position/hpath")
        self.airspeed_ref = xp.findDataRef("sim/flightmodel/position/indicated_airspeed")
        self.vertical_speed_ref = xp.findDataRef("sim/flightmodel/position/vh_ind")
        self.elevator_ref = xp.findDataRef("sim/joystick/yoke_pitch_ratio")
        self.throttle_ref = xp.findDataRef("sim/flightmodel/engine/ENGN_thro")
        self.aileron_ref = xp.findDataRef("sim/joystick/yoke_roll_ratio")
        self.gear_ref = xp.findDataRef("sim/aircraft/parts/acf_gear_deploy")
        self.flaps_ref = xp.findDataRef("sim/flightmodel/controls/flaprqst")

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
            self.window_id = xp.createWindowEx(50, 700, 600, 500, 1,
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
            for message in self.debug_messages[-16:]:
                xp.drawString(color, left + 5, y_pos, message, 0, xp.Font_Basic)
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
            self.debug_messages = []  # Clear previous messages
            state = self.read_state()
            labels = [
                "Distance to threshold (m)",
                "Lateral deviation (m)",
                "Height above runway (ft)",   
                "Heading deviation (deg)",
                "Airspeed (kt)",
                "Vertical speed (ft/min)",
                "Pitch (deg)",
                "Bank (deg)",
                "Elevator pos",
                "Throttle pos",
                "Gear pos",
                "Flap pos"
            ]
            for label, value in zip(labels, state):
                self.add_debug_message(f"{label}: {value:.3f}")
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

    def read_state(self):
        # Get aircraft position and attitude
        lat = xp.getDataf(self.lat_ref)
        lon = xp.getDataf(self.lon_ref)
        elev = xp.getDataf(self.elevation_ref)  # meters MSL
        pitch = xp.getDataf(self.pitch_ref)  # deg
        bank = xp.getDataf(self.bank_ref)    # deg
        heading = xp.getDataf(self.heading_ref)  # deg true

        # Flight parameters
        airspeed = xp.getDataf(self.airspeed_ref)  # knots
        vertical_speed = xp.getDataf(self.vertical_speed_ref)  # ft/min

        # Controls
        elevator = xp.getDataf(self.elevator_ref)
        # Throttle (array)
        throttle_arr = [0.0] * 8
        xp.getDatavf(self.throttle_ref, throttle_arr, 0, 1)
        throttle = throttle_arr[0]
        aileron = xp.getDataf(self.aileron_ref)

        # Gear/Flaps (gear is array)
        gear_arr = [0.0] * 10
        xp.getDatavf(self.gear_ref, gear_arr, 0, 1)
        gear = gear_arr[0]
        flaps = xp.getDataf(self.flaps_ref)

        # Calculate runway-relative values
        R = 6371000  # Earth radius in meters
        dlat = radians(self.RUNWAY_LAT - lat)
        dlon = radians(self.RUNWAY_LON - lon)
        a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(self.RUNWAY_LAT)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance_to_threshold = R * c

        # Heading to runway
        y = sin(radians(self.RUNWAY_LON - lon)) * cos(radians(self.RUNWAY_LAT))
        x = cos(radians(lat)) * sin(radians(self.RUNWAY_LAT)) - sin(radians(lat)) * cos(radians(self.RUNWAY_LAT)) * cos(radians(self.RUNWAY_LON - lon))
        bearing = (atan2(y, x) * 180.0 / 3.141592653589793 + 360.0) % 360.0
        heading_deviation = ((heading - self.RUNWAY_HEADING + 180) % 360) - 180  # signed deviation

        # Lateral deviation (meters, perpendicular to runway centerline)
        lateral_deviation = distance_to_threshold * sin(radians(bearing - self.RUNWAY_HEADING))

        # Height above runway (feet)
        height_above_runway = elev * 3.28084  # meters to feet

        # Compose state vector
        state = [
            distance_to_threshold,      # meters
            lateral_deviation,         # meters
            height_above_runway,       # feet
            heading_deviation,         # degrees
            airspeed,                  # knots
            vertical_speed,            # ft/min
            pitch,                     # degrees
            bank,                      # degrees
            elevator,                  # -1 to 1
            throttle,                  # 0 to 1
            gear,                      # 0/1
            flaps                      # handle position
        ]
        return state

    def set_actions(self, elevator=None, throttle=None, aileron=None):
        if elevator is not None:
            xp.setDataf(self.elevator_ref, elevator)
        if throttle is not None:
            throttle_arr = [0.0] * 8
            throttle_arr[0] = throttle
            xp.setDatavf(self.throttle_ref, throttle_arr, 0, 1)
        if aileron is not None:
            xp.setDataf(self.aileron_ref, aileron)