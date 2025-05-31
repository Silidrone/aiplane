from XPPython3 import xp
from XPPython3.utils.easy_python import EasyPython
import time
import numpy as np
from math import radians, sin, cos, sqrt, atan2, tan

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

        # LTBJ 34R
        self.RUNWAY_LAT = 38.278404
        self.RUNWAY_LON = 27.161163
        self.RUNWAY_HEADING = 346.573  # degrees
        self.RUNWAY_ELEVATION = 114.028  # meters

        # Cache all DataRefs used in state/action
        self.lat_ref = xp.findDataRef("sim/flightmodel/position/latitude")
        self.lon_ref = xp.findDataRef("sim/flightmodel/position/longitude")
        self.elevation_ref = xp.findDataRef("sim/flightmodel/position/elevation")
        self.pitch_ref = xp.findDataRef("sim/flightmodel/position/theta")
        self.bank_ref = xp.findDataRef("sim/flightmodel/position/phi")
        self.heading_ref = xp.findDataRef("sim/flightmodel/position/magpsi")  # Use magnetic heading
        self.magpsi_ref = xp.findDataRef("sim/flightmodel/position/magpsi")
        self.truepsi_ref = xp.findDataRef("sim/flightmodel/position/psi")
        self.airspeed_ref = xp.findDataRef("sim/flightmodel/position/indicated_airspeed")
        self.vertical_speed_ref = xp.findDataRef("sim/flightmodel/position/vh_ind")
        self.elevator_ref = xp.findDataRef("sim/joystick/yoke_pitch_ratio")
        self.throttle_ref = xp.findDataRef("sim/flightmodel/engine/ENGN_thro")
        self.aileron_ref = xp.findDataRef("sim/joystick/yoke_roll_ratio")
        self.flaps_ref = xp.findDataRef("sim/flightmodel/controls/flaprqst")
        self.rudder_ref = xp.findDataRef("sim/joystick/yoke_heading_ratio")
        self.alpha_ref = xp.findDataRef("sim/flightmodel/position/alpha")
        self.beta_ref = xp.findDataRef("sim/flightmodel/position/beta")
        self.vpath_ref = xp.findDataRef("sim/flightmodel/position/vpath")
        self.hpath_ref = xp.findDataRef("sim/flightmodel/position/hpath")

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
                "Distance to runway (m)",
                "MSL (m)",
                "Lateral deviation (m)",
                "Vertical deviation (m)",
                "Heading deviation (deg)",
                "Vertical speed (ft/min)",
                "Pitch (deg)",
                "Bank (deg)",   
            ]

            # Normalization parameters
            norm_params = [
                (0, 6000),        # Distance
                (114.028, 450),   # MSL
                (-750, 750),      # Lateral deviation
                (-20, 50),        # Vertical deviation
                (-10, 10),        # Heading deviation
                (-10, 10),        # Vertical speed
                (-20, 20),        # Pitch
                (-25, 25),        # Bank
            ]

            for label, raw, norm in zip(labels, state, norm_params):
                if norm is not None:
                    min_v, max_v = norm
                    normalized = max(0.0, min(1.0, (raw - min_v) / (max_v - min_v)))
                    self.add_debug_message(f"{label}: {raw:.3f} (normalized: {normalized:.3f})")
                else:
                    self.add_debug_message(f"{label}: {raw:.3f} (normalized: {raw:.3f})")
        except Exception as e:
            self.add_debug_message(f"READ ERROR: {e}")

    def reset_to_approach(self):
        try:
            # xp.loadDataFile(xp.DataFile_Situation, 'Output/cessna.sit')
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
            self.set_actions(elevator=0.0, throttle=0.3, aileron=0.0, flaps=1.0)
            
            # Set rudder to a fixed value to counteract the Cessna left engine bias
            # xp.setDataf(self.rudder_ref, 0.3)
            self.episode_count += 1
        except Exception as e:
            self.add_debug_message(f"RESET ERROR: {e}")

    def read_state(self):
        # Get aircraft position and attitude
        lat = xp.getDataf(self.lat_ref)
        lon = xp.getDataf(self.lon_ref)
        msl = xp.getDataf(self.elevation_ref)  # meters above ground
        if msl < 0:
            msl = 0.0
        pitch = xp.getDataf(self.pitch_ref)  # deg
        bank = xp.getDataf(self.bank_ref)    # deg
        truepsi = xp.getDataf(self.truepsi_ref)

        # Flight parameters
        vertical_speed = xp.getDataf(self.vertical_speed_ref)  # ft/min

        state = [
            self.haversine_distance(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON),     # meters (distance to runway threshold)
            msl,       # meters above ground
            self.lateral_deviation(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON, truepsi),  
            self.vertical_deviation(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON, msl, self.RUNWAY_ELEVATION),
            self.RUNWAY_HEADING - truepsi,         # heading deviation in degrees (true)
            vertical_speed,            # ft/min
            pitch,                     # degrees
            bank,                      # degrees
        ]
        return state

    def set_actions(self, elevator=None, throttle=None, aileron=None, flaps=None):
        if elevator is not None:
            xp.setDataf(self.elevator_ref, elevator)
        if throttle is not None:
            throttle_arr = [0.0] * 8
            throttle_arr[0] = throttle
            xp.setDatavf(self.throttle_ref, throttle_arr, 0, 1)
        if aileron is not None:
            xp.setDataf(self.aileron_ref, aileron)
        if flaps is not None:
            xp.setDataf(self.flaps_ref, flaps)

    def lateral_deviation(self, from_lat, from_lon, to_lat, to_lon, true_psi):
        from math import radians, sin, cos
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * cos(radians(to_lat))
        d_lat = from_lat - to_lat
        d_lon = from_lon - to_lon
        north = d_lat * meters_per_deg_lat
        east = d_lon * meters_per_deg_lon
        truepsi_rad = radians(true_psi)
        truepsi_east = sin(truepsi_rad)
        truepsi_north = cos(truepsi_rad)
        latdev_truepsi = east * truepsi_north - north * truepsi_east
        return latdev_truepsi
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1 = radians(lat1)
        phi2 = radians(lat2)
        d_phi = radians(lat2 - lat1)
        d_lambda = radians(lon2 - lon1)

        a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def vertical_deviation(self, from_lat, from_lon, to_lat, to_lon, aircraft_elev, runway_elev, glide_slope_deg=3.0):
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * cos(radians(to_lat))
        d_lat = from_lat - to_lat
        d_lon = from_lon - to_lon
        north = d_lat * meters_per_deg_lat
        east = d_lon * meters_per_deg_lon
        ground_dist = (north ** 2 + east ** 2) ** 0.5
        glide_slope_rad = radians(glide_slope_deg)
        ideal_alt = tan(glide_slope_rad) * ground_dist + runway_elev
        vert_dev = aircraft_elev - ideal_alt
        return vert_dev
