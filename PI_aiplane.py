from XPPython3 import xp
from XPPython3.utils.easy_python import EasyPython
import time
import torch
import os
from aiplane_util import lateral_deviation, haversine_distance, vertical_deviation, draw_window_callback
from sarsa import SARSA
from policy import EpsilonGreedyPolicy
from value_strategy import TorchValueStrategy
from aiplane_models import AiplaneQNet, feature_extractor
from aiplane import AiplaneEnv

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
        
        # RL Training components
        self.rl_enabled = False
        self.environment = None
        self.model = None
        self.policy = None
        self.value_strategy = None
        self.sarsa_solver = None
        
        # Training parameters
        self.learning_rate = 0.001
        self.epsilon = 0.3
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.0001
        self.discount_rate = 0.95
        self.hidden_size = 128
        self.n_episodes = 1000
        self.model_path = os.path.join(os.path.dirname(__file__), "aiplane_model.pth")

        # LTBJ 34R
        self.RUNWAY_LAT = 38.278404
        self.RUNWAY_LON = 27.161163
        self.RUNWAY_HEADING = 346.573  # degrees
        self.RUNWAY_ELEVATION = 114.028  # meters
    
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
            # Wrap the util draw_window_callback to provide instance variables
            def window_draw_cb(inWindowID, inRefcon):
                draw_window_callback(inWindowID, inRefcon, self.debug_messages, self.episode_count, xp)
            self.window_id = xp.createWindowEx(50, 700, 600, 500, 1,
                                             window_draw_cb,
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
            
            # Training status
            if self.rl_enabled:
                epsilon = f"ε: {self.policy.epsilon:.3f}" if self.policy else "ε: N/A"
                xp.drawString(color, left + 5, y_pos, f"RL: READY | {epsilon}", 0, xp.Font_Basic)
                y_pos -= 15
            
            # Instructions
            xp.drawString(color, left + 5, y_pos, "Left click: Reset | Right click: Start Training", 0, xp.Font_Basic)
            y_pos -= 15
            
            # Debug messages
            for message in self.debug_messages[-13:]:
                xp.drawString(color, left + 5, y_pos, message, 0, xp.Font_Basic)
                y_pos -= 12
                
        except Exception as e:
            pass
        
    def mouse_click_callback(self, inWindowID, x, y, inMouse, inRefcon):
        if inMouse == xp.MouseDown:
            self.start_training()
            # self.reset_to_approach()

        return 1
        
    def key_callback(self, inWindowID, inKey, inFlags, inVirtualKey, inRefcon, losingFocus):
        pass
        
    def cursor_callback(self, inWindowID, x, y, inRefcon):
        return xp.CursorDefault

    def onStart(self):
        # Cache all DataRefs used in state/action
        self.lat_ref = xp.findDataRef("sim/flightmodel/position/latitude")
        self.lon_ref = xp.findDataRef("sim/flightmodel/position/longitude")
        self.elevation_ref = xp.findDataRef("sim/flightmodel/position/elevation")
        self.pitch_ref = xp.findDataRef("sim/flightmodel/position/theta")
        self.bank_ref = xp.findDataRef("sim/flightmodel/position/phi")
        self.heading_ref = xp.findDataRef("sim/flightmodel/position/magpsi")  # Use magnetic heading
        self.magpsi_ref = xp.findDataRef("sim/flightmodel/position/magpsi")
        self.truepsi_ref = xp.findDataRef("sim/flightmodel/position/psi")
        self.airspeed_ref = xp.findDataRef("sim/flightmodel/position/true_airspeed")
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
        self.initialize_rl_system()

    def after_physics(self):
        try:
            current_time = time.time()
            
            # Print current state every frame
            # self.print_current_state()
            
            # Clear debug messages every 10 seconds
            if current_time - self.last_clear_time >= 10.0:
                self.clear_debug_messages()
                
            # Run one training step per frame
            self.step_training()
        except Exception as e:
            self.add_debug_message(f"Error: {e}")

    def print_current_state(self):
        try:
            from aiplane_models import feature_extractor
            
            self.debug_messages = []  # Clear previous messages
            state = self.read_state()
            
            # Use dummy action for feature extraction
            dummy_action = (0.0, 0.5, 0.0, 0.0)  # elevator, throttle, aileron, flaps
            features = feature_extractor(state, dummy_action).squeeze().tolist()
            
            labels = [
                "Distance to runway (m)",
                "MSL (m)", 
                "Lateral deviation (m)",
                "Vertical deviation (m)",
                "Heading deviation (deg)",
                "Vertical speed (ft/min)",
                "Pitch (deg)",
                "Bank (deg)",
                "Airspeed (knots)",   
            ]

            for i, (label, raw) in enumerate(zip(labels, state)):
                normalized = features[i]
                self.add_debug_message(f"{label}: {raw:.3f} (normalized: {normalized:.3f})")
        except Exception as e:
            self.add_debug_message(f"READ ERROR: {e}")

    def reset_to_approach(self):
        try:
            self.add_debug_message(f"RESETTING: {self.episode_count}#{self.steps_in_episode}")
            print(f"RESETTING: {self.episode_count}#{self.steps_in_episode}")
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
        airspeed = xp.getDataf(self.airspeed_ref)  # knots

        state = [
            haversine_distance(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON),     # meters (distance to runway threshold)
            msl,       # meters above ground
            lateral_deviation(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON, truepsi),  
            vertical_deviation(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON, msl, self.RUNWAY_ELEVATION),
            self.RUNWAY_HEADING - truepsi,         # heading deviation in degrees (true)
            vertical_speed,            # ft/min
            pitch,                     # degrees
            bank,                      # degrees
            airspeed,                  # knots
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
    
    def initialize_rl_system(self):
        try:
            self.environment = AiplaneEnv(self)
            self.environment.initialize()
            
            self.add_debug_message(f"PyTorch version: {torch.__version__}")
            self.add_debug_message(f"CUDA built: {torch.version.cuda}")
            self.add_debug_message(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                gpu_name = torch.cuda.get_device_name(0)
                self.add_debug_message(f"GPU: {gpu_name}")
                self.add_debug_message(f"Using device: {device}")
            else:
                device = torch.device("cpu")
                self.add_debug_message("CUDA not available, using CPU")
                self.add_debug_message(f"Using device: {device}")
            
            input_size = 13  # 9 state features + 4 action features
            self.model = AiplaneQNet(input_size, self.hidden_size)
            self.model.to(device)
            
            if os.path.exists(self.model_path):
                try:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                    self.add_debug_message("Loaded existing model")
                except Exception as e:
                    self.add_debug_message(f"Could not load model: {e}")
            
            self.value_strategy = TorchValueStrategy(self.model, feature_extractor, self.learning_rate, device)
            self.value_strategy.initialize(self.environment)
            
            self.policy = EpsilonGreedyPolicy(self.value_strategy, self.epsilon, self.min_epsilon, self.epsilon_decay)
            self.policy.initialize(self.environment, self.value_strategy)
            
            self.sarsa_solver = SARSA(self.environment, self.policy, self.value_strategy, self.discount_rate, self.n_episodes, True)
            
            self.rl_enabled = True
            self.add_debug_message("RL system initialized successfully")
            
        except Exception as e:
            self.add_debug_message(f"RL init error: {e}")
            self.rl_enabled = False
    
    def start_training(self):
        if not self.rl_enabled:
            self.add_debug_message("RL system not initialized")
            return
            
        if not hasattr(self, 'training_active'):
            self.training_active = True
            self.current_episode = 0
            self.max_episodes = 1000
            self.add_debug_message("Starting SARSA training...")
            self.training_state = None
            self.training_action = None
            self.steps_in_episode = 0
            
    def step_training(self):
        """Run one training step per X-Plane frame to keep simulator responsive"""
        if not hasattr(self, 'training_active') or not self.training_active:
            return
            
        try:
            if self.current_episode >= self.max_episodes:
                self.training_active = False
                self.save_model()
                self.add_debug_message("Training completed!")
                return
                
            # Start new episode
            if self.training_state is None:
                self.current_episode += 1
                self.training_state = self.environment.reset()
                self.training_action = self.policy.sample(self.training_state)
                self.steps_in_episode = 0
                if self.current_episode % 100 == 0:
                    self.add_debug_message(f"Episode {self.current_episode}/{self.max_episodes}")
                    
            # Take one step
            new_state, reward = self.environment.step(self.training_state, self.training_action)
            new_action = self.policy.sample(new_state)
            
            # Print action taken
            elevator, throttle, aileron, flaps = self.training_action
            self.add_debug_message(f"Action: E:{elevator:.2f} T:{throttle:.2f} A:{aileron:.2f} F:{flaps:.2f} R:{reward:.3f}")
            
            # SARSA update: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            if self.environment.is_terminal(new_state):
                target_q = reward
            else:
                next_q = self.value_strategy.Q(new_state, new_action)
                target_q = reward + self.discount_rate * next_q
            
            self.value_strategy.update(self.training_state, self.training_action, target_q)
            
            self.training_state = new_state
            self.training_action = new_action
            self.steps_in_episode += 1
            
            # Check if episode is done
            if self.environment.is_terminal(new_state) or self.steps_in_episode > 1000:
                self.training_state = None
                
        except Exception as e:
            self.add_debug_message(f"Training error: {e}")
            self.training_active = False
            self.save_model()
    
    def save_model(self):
        if self.model is not None:
            try:
                torch.save(self.model.state_dict(), self.model_path)
                self.add_debug_message("Model saved")
            except Exception as e:
                self.add_debug_message(f"Save error: {e}")
