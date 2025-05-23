from XPPython3 import xp
import math
import time

class PythonInterface:
    def __init__(self):
        self.Name = "Airplane RL Landing Plugin"
        self.Sig = "com.example.airplane_rl"
        self.Desc = "Reinforcement Learning environment for landing training"
        
        # Cached data references for performance
        self.altitude_ref = None
        self.vertical_speed_ref = None
        self.airspeed_ref = None
        self.pitch_ref = None
        self.roll_ref = None
        self.heading_ref = None
        self.latitude_ref = None
        self.longitude_ref = None
        self.flaps_ref = None
        self.gear_ref = None
        self.groundspeed_ref = None
        
        # Control datarefs
        self.yoke_pitch_ref = None
        self.yoke_roll_ref = None
        self.yoke_heading_ref = None
        self.throttle_ref = None
        self.flap_control_ref = None
        self.gear_control_ref = None
        
        # Cessna 172 specific datarefs
        self.mixture_ref = None
        self.carb_heat_ref = None
        self.elevator_trim_ref = None
        self.rpm_ref = None
        
        # Landing constants for Cessna 172
        self.APPROACH_SPEED = 65      # knots (final approach with full flaps)
        self.PATTERN_SPEED = 80       # knots (downwind/base)
        self.FULL_FLAP_SPEED = 85     # knots (maximum speed for full flaps)
        self.STALL_SPEED_CLEAN = 54   # knots (flaps up)
        self.STALL_SPEED_LANDING = 47 # knots (full flaps)
        
        self.flight_loop_id = None
        self.window_id = None
        self.episode_count = 0
        self.last_state = {}
        self.status_message = "Starting..."
        
        # Debug message system with timer
        self.debug_messages = []
        self.max_debug_messages = 8
        self.last_clear_time = time.time()
        self.last_status_update = time.time()

    def add_debug_message(self, message):
        """Add a debug message to be displayed in window"""
        self.debug_messages.append(message)
        if len(self.debug_messages) > self.max_debug_messages:
            self.debug_messages.pop(0)  # Remove oldest message

    def clear_debug_messages(self):
        """Clear all debug messages"""
        self.debug_messages = []
        current_time = time.time()
        self.add_debug_message(f"--- Debug cleared at {current_time:.0f} ---")
        self.last_clear_time = current_time

    def XPluginStart(self):
        try:
            self.add_debug_message("=== STARTING PLUGIN ===")
            
            # Cache all commonly used datarefs for performance
            self.add_debug_message("Finding datarefs...")
            self.altitude_ref = xp.findDataRef("sim/flightmodel/position/y_agl")
            self.vertical_speed_ref = xp.findDataRef("sim/flightmodel/position/vh_ind")
            self.airspeed_ref = xp.findDataRef("sim/flightmodel/position/indicated_airspeed")
            self.pitch_ref = xp.findDataRef("sim/flightmodel/position/theta")
            self.roll_ref = xp.findDataRef("sim/flightmodel/position/phi")
            self.heading_ref = xp.findDataRef("sim/flightmodel/position/psi")
            self.latitude_ref = xp.findDataRef("sim/flightmodel/position/latitude")
            self.longitude_ref = xp.findDataRef("sim/flightmodel/position/longitude")
            self.flaps_ref = xp.findDataRef("sim/flightmodel/controls/flaprqst")
            self.gear_ref = xp.findDataRef("sim/aircraft/parts/acf_gear_deploy")
            self.groundspeed_ref = xp.findDataRef("sim/flightmodel/position/groundspeed")
            
            # Control datarefs
            self.yoke_pitch_ref = xp.findDataRef("sim/cockpit2/controls/yoke_pitch_ratio")
            self.yoke_roll_ref = xp.findDataRef("sim/cockpit2/controls/yoke_roll_ratio")
            self.yoke_heading_ref = xp.findDataRef("sim/cockpit2/controls/yoke_heading_ratio")
            self.throttle_ref = xp.findDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all")
            self.flap_control_ref = xp.findDataRef("sim/cockpit2/controls/flap_ratio")
            self.gear_control_ref = xp.findDataRef("sim/cockpit2/controls/gear_handle_down")
            
            # Cessna 172 specific datarefs
            self.mixture_ref = xp.findDataRef("sim/cockpit2/engine/actuators/mixture_ratio_all")
            self.carb_heat_ref = xp.findDataRef("sim/cockpit2/engine/actuators/carb_heat_ratio")
            self.elevator_trim_ref = xp.findDataRef("sim/cockpit2/controls/elevator_trim")
            self.rpm_ref = xp.findDataRef("sim/cockpit2/engine/indicators/engine_speed_rpm_all")
            self.add_debug_message("All datarefs found!")
            
            # Set simulation speed (can be adjusted for training)
            xp.setDataf(xp.findDataRef("sim/time/sim_speed"), 2.0)
            self.add_debug_message("Sim speed set to 2x")
            
            # Create window immediately for visibility
            self.create_display_window()
            self.add_debug_message("Window created!")
            self.status_message = "Plugin Started Successfully"
            
            return self.Name, self.Sig, self.Desc
            
        except Exception as e:
            self.add_debug_message(f"ERROR in Start: {e}")
            self.status_message = f"ERROR in Start: {e}"
            return self.Name, self.Sig, self.Desc

    def XPluginStop(self):
        if self.flight_loop_id:
            xp.destroyFlightLoop(self.flight_loop_id)
        if self.window_id:
            xp.destroyWindow(self.window_id)

    def XPluginEnable(self):
        try:
            self.add_debug_message("=== ENABLING PLUGIN ===")
            # Register flight loop callback
            self.flight_loop_id = xp.createFlightLoop(self.flight_loop_callback)
            xp.scheduleFlightLoop(self.flight_loop_id, -1)
            self.add_debug_message("Flight loop registered!")
            self.status_message = "Plugin Enabled Successfully"
            return 1
            
        except Exception as e:
            self.add_debug_message(f"ERROR in Enable: {e}")
            self.status_message = f"ERROR in Enable: {e}"
            return 0

    def XPluginDisable(self):
        if self.flight_loop_id:
            xp.scheduleFlightLoop(self.flight_loop_id, 0)
        if self.window_id:
            xp.destroyWindow(self.window_id)

    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        pass

    def create_display_window(self):
        """Create a window for real-time state display"""
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
        """Draw real-time aircraft state information"""
        try:
            left, top, right, bottom = xp.getWindowGeometry(inWindowID)
            
            # Draw translucent background
            xp.drawTranslucentDarkBox(left, top, right, bottom)
            
            color = (1.0, 1.0, 1.0)  # White text
            y_pos = top - 15
            
            # Title
            xp.drawString(color, left + 5, y_pos, "Cessna 172 - LTBJ Landing RL", 0, xp.Font_Basic)
            y_pos -= 15
            
            # Status message
            xp.drawString(color, left + 5, y_pos, f"Status: {self.status_message}", 0, xp.Font_Basic)
            y_pos -= 15
            
            # Display current state
            try:
                state = self.get_state()
                
                # Key flight parameters
                xp.drawString(color, left + 5, y_pos, f"Alt: {state['altitude']:.0f}ft AGL  Speed: {state['indicated_airspeed']:.0f}kt  VS: {state['vertical_speed']:.0f}fpm", 0, xp.Font_Basic)
                y_pos -= 15
                
                xp.drawString(color, left + 5, y_pos, f"Dist to RWY: {state['distance_to_runway']:.1f}nm  Glideslope: {state['glideslope_deviation']:.0f}ft", 0, xp.Font_Basic)
                y_pos -= 15
                
                xp.drawString(color, left + 5, y_pos, f"Pitch: {math.degrees(state['pitch']):.1f}°  Flaps: {state['flaps_position']:.2f}  RPM: {state['rpm']:.0f}", 0, xp.Font_Basic)
                y_pos -= 15
                
                xp.drawString(color, left + 5, y_pos, f"Episodes: {self.episode_count}  --  CLICK WINDOW TO RESET AIRCRAFT", 0, xp.Font_Basic)
                y_pos -= 20
                
            except Exception as e:
                xp.drawString(color, left + 5, y_pos, f"Error getting state: {str(e)[:50]}", 0, xp.Font_Basic)
                y_pos -= 15
                
            # Draw debug messages section with clear countdown
            current_time = time.time()
            time_since_clear = current_time - self.last_clear_time
            time_until_clear = 10.0 - time_since_clear
            
            if time_until_clear <= 0:
                clear_text = "=== DEBUG MESSAGES - CLEARING NOW! ==="
            else:
                clear_text = f"=== DEBUG MESSAGES - CLEAR in {time_until_clear:.1f}s ==="
            
            xp.drawString(color, left + 5, y_pos, clear_text, 0, xp.Font_Basic)
            y_pos -= 15
            
            # Show recent debug messages
            for message in self.debug_messages[-8:]:  # Show last 8 messages
                xp.drawString(color, left + 5, y_pos, message[:70], 0, xp.Font_Basic)  # Truncate long messages
                y_pos -= 12
                
        except Exception as e:
            pass  # Can't show error in window if window drawing fails
        
    def mouse_click_callback(self, inWindowID, x, y, inMouse, inRefcon):
        """Handle mouse clicks on the window"""
        if inMouse == xp.MouseDown:
            self.add_debug_message("*** MOUSE CLICKED! Starting reset... ***")
            # Reset aircraft on mouse click for testing
            self.reset_to_approach()
            self.episode_count += 1
            self.add_debug_message(f"*** RESET COMPLETE! Episode #{self.episode_count} ***")
            self.status_message = f"Reset #{self.episode_count} completed"
        return 1
        
    def key_callback(self, inWindowID, inKey, inFlags, inVirtualKey, inRefcon, losingFocus):
        pass
        
    def cursor_callback(self, inWindowID, x, y, inRefcon):
        return xp.CursorDefault

    def flight_loop_callback(self, sinceLastCall, sinceLastFlightLoop, counter, refcon):
        # Store current state for display
        try:
            self.last_state = self.get_state()
            
            current_time = time.time()
            
            # Clear debug messages every 10 seconds using real time
            if current_time - self.last_clear_time >= 10.0:
                self.clear_debug_messages()
            
            # Update status message every 5 seconds
            if current_time - self.last_status_update >= 5.0:
                alt = self.last_state['altitude']
                speed = self.last_state['indicated_airspeed']
                dist = self.last_state['distance_to_runway']
                self.status_message = f"Flying - Alt:{alt:.0f} Spd:{speed:.0f} Dist:{dist:.1f}"
                self.add_debug_message(f"Update: Alt:{alt:.0f} Spd:{speed:.0f} Dist:{dist:.1f}")
                self.last_status_update = current_time
                
        except Exception as e:
            self.add_debug_message(f"Flight loop error: {e}")
            self.status_message = f"Flight loop error: {e}"
            
        return -1  # Continue calling
    
    def get_state(self):
        try:
            altitude = xp.getDataf(self.altitude_ref)
            
            return {
                # Aircraft position & orientation
                'altitude': altitude,
                'vertical_speed': xp.getDataf(self.vertical_speed_ref) if self.vertical_speed_ref else 0,
                'indicated_airspeed': xp.getDataf(self.airspeed_ref) if self.airspeed_ref else 0,
                'pitch': xp.getDataf(self.pitch_ref) if self.pitch_ref else 0,
                'roll': xp.getDataf(self.roll_ref) if self.roll_ref else 0,
                'heading': xp.getDataf(self.heading_ref) if self.heading_ref else 0,
                
                # Landing-specific
                'distance_to_runway': self.calculate_runway_distance(),
                'glideslope_deviation': self.calculate_glideslope_deviation(),
                'localizer_deviation': 0.0,   # Placeholder for now
                
                # Aircraft configuration
                'flaps_position': xp.getDataf(self.flaps_ref) if self.flaps_ref else 0,
                'gear_position': xp.getDataf(self.gear_ref) if self.gear_ref else 1.0,  # Always 1.0 for fixed gear
                'ground_speed': xp.getDataf(self.groundspeed_ref) if self.groundspeed_ref else 0,
                
                # Cessna 172 specific
                'rpm': xp.getDataf(self.rpm_ref) if self.rpm_ref else 0,
                'mixture': xp.getDataf(self.mixture_ref) if self.mixture_ref else 0,
                'carb_heat': xp.getDataf(self.carb_heat_ref) if self.carb_heat_ref else 0,
                'elevator_trim': xp.getDataf(self.elevator_trim_ref) if self.elevator_trim_ref else 0
            }
        except Exception as e:
            self.add_debug_message(f"get_state error: {e}")
            return {'altitude': 0, 'indicated_airspeed': 0, 'distance_to_runway': 0, 'vertical_speed': 0}
    
    def calculate_runway_distance(self):
        # İzmir Adnan Menderes Airport (LTBJ) runway coordinates
        aircraft_lat = xp.getDatad(self.latitude_ref)
        aircraft_lon = xp.getDatad(self.longitude_ref)
        
        # LTBJ Airport coordinates: 38°17'35"N, 27°9'30"E (38.2897°N, 27.1583°E)
        # Primary runway 16L/34R threshold coordinates
        runway_lat = 38.2897
        runway_lon = 27.1583
        
        # Simple distance calculation
        lat_diff = aircraft_lat - runway_lat
        lon_diff = aircraft_lon - runway_lon
        distance_degrees = math.sqrt(lat_diff*lat_diff + lon_diff*lon_diff)
        distance_nm = distance_degrees * 60  # Rough conversion to nautical miles
        
        return distance_nm

    def calculate_glideslope_deviation(self):
        """Calculate glideslope deviation for 3° approach to LTBJ Runway 16L"""
        altitude_agl = xp.getDataf(self.altitude_ref)
        distance_nm = self.calculate_runway_distance()
        
        # Standard 3° glideslope: 318 feet per nautical mile
        target_altitude = distance_nm * 318
        
        # Deviation in feet (positive = above glideslope, negative = below)
        glideslope_deviation = altitude_agl - target_altitude
        
        return glideslope_deviation
        
    def apply_action(self, action):
        # Apply actions using cached datarefs
        xp.setDataf(self.yoke_pitch_ref, action.get('elevator', 0.0))
        xp.setDataf(self.yoke_roll_ref, action.get('aileron', 0.0))
        xp.setDataf(self.yoke_heading_ref, action.get('rudder', 0.0))
        xp.setDataf(self.throttle_ref, action.get('throttle', 0.4))
        
        if 'flaps' in action:
            xp.setDataf(self.flap_control_ref, action['flaps'])
        if 'gear' in action:
            xp.setDataf(self.gear_control_ref, action['gear'])
    
    def reset_to_approach(self):
        try:
            self.add_debug_message("RESET: Starting reset...")
            
            xplane_path = xp.getSystemPath()
            self.add_debug_message(f"RESET: X-Plane system path: {xplane_path}")
        except Exception as e:
            self.add_debug_message(f"RESET ERROR: {e}")
