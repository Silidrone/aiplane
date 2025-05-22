import xp
from xppython3 import xplane_plugin

class AirplaneLearningPlugin(xplane_plugin.Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Airplane RL Landing Plugin"
        
        self.altitude_ref = xp.findDataRef("sim/flightmodel/position/y_agl")
        xp.setDataf(xp.findDataRef("sim/time/sim_speed"), 2.0)  # 2x speedup
        
    def enable(self):
        # Register flight loop callback
        self.flight_loop_id = xp.createFlightLoop(self.flight_loop_callback)
        xp.scheduleFlightLoop(self.flight_loop_id, -1)
        return 1
        
    def flight_loop_callback(self, elapsed_since_last_call, elapsed_since_last_flight_loop, counter):
        # Environment logic here
        print("flight_loop_callback called!")
        return -1  # Negative value means "call me next frame"
    
    def get_state(self):
        return {
            # Aircraft position & orientation
            'altitude': xp.getDataf(self.altitude_ref),
            'vertical_speed': xp.getDataf(xp.findDataRef("sim/flightmodel/position/vh_ind")),
            'indicated_airspeed': xp.getDataf(xp.findDataRef("sim/flightmodel/position/indicated_airspeed")),
            'pitch': xp.getDataf(xp.findDataRef("sim/flightmodel/position/theta")),
            'roll': xp.getDataf(xp.findDataRef("sim/flightmodel/position/phi")),
            'heading': xp.getDataf(xp.findDataRef("sim/flightmodel/position/psi")),
            
            # Landing-specific
            'distance_to_runway': self.calculate_runway_distance(),
            'glideslope_deviation': self.calculate_glideslope_deviation(),
            'localizer_deviation': self.calculate_localizer_deviation(),
            
            # Aircraft configuration
            'flaps_position': xp.getDataf(xp.findDataRef("sim/flightmodel/controls/flaprqst")),
            'gear_position': xp.getDataf(xp.findDataRef("sim/aircraft/parts/acf_gear_deploy")),
            'ground_speed': xp.getDataf(xp.findDataRef("sim/flightmodel/position/groundspeed"))
        }
        
    def apply_action(self, action):
        # Apply actions using datarefs
        xp.setDataf(xp.findDataRef("sim/cockpit2/controls/yoke_pitch_ratio"), action['elevator'])
        xp.setDataf(xp.findDataRef("sim/cockpit2/controls/yoke_roll_ratio"), action['aileron'])
        xp.setDataf(xp.findDataRef("sim/cockpit2/controls/yoke_heading_ratio"), action['rudder'])
        xp.setDataf(xp.findDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all"), action['throttle'])
        
        if 'flaps' in action:
            xp.setDataf(xp.findDataRef("sim/cockpit2/controls/flap_ratio"), action['flaps'])
        if 'gear' in action:
            xp.setDataf(xp.findDataRef("sim/cockpit2/controls/gear_handle_down"), action['gear'])
    
    def reset_to_approach(self):
        # Override physics
        xp.setDatai(xp.findDataRef("sim/operation/override/override_planepath"), 1)
        
        lat = 41.0  # Latitude
        lon = 28.0  # Longitude
        alt = 1500  # Altitude (feet AGL)
        heading = 270  # Runway heading
        
        # Set position
        xp.setDatad(xp.findDataRef("sim/flightmodel/position/latitude"), lat)
        xp.setDatad(xp.findDataRef("sim/flightmodel/position/longitude"), lon)
        xp.setDataf(xp.findDataRef("sim/flightmodel/position/elevation"), alt)
        xp.setDataf(xp.findDataRef("sim/flightmodel/position/psi"), heading)
        
        # Set initial velocity (airspeed for approach)
        speed = 160  # knots
        xp.setDataf(xp.findDataRef("sim/flightmodel/position/indicated_airspeed"), speed)
        
        # Configure aircraft for approach
        xp.setDataf(xp.findDataRef("sim/cockpit2/controls/flap_ratio"), 0.33)  # Approach flaps
        xp.setDataf(xp.findDataRef("sim/cockpit2/controls/gear_handle_down"), 1.0)  # Gear down
        xp.setDataf(xp.findDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all"), 0.4)  # Initial throttle
        
        # Disable physics override
        xp.setDatai(xp.findDataRef("sim/operation/override/override_planepath"), 0)
