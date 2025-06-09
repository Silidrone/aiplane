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
        
        self.debug_messages = []
        self.max_debug_messages = 16
        self.last_clear_time = time.time()
        
        self.rl_enabled = False
        self.environment = None
        self.model = None
        self.policy = None
        self.value_strategy = None
        self.sarsa_solver = None
        
        self.learning_rate = 0.001
        self.epsilon = 0.3
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.0001
        self.discount_rate = 1
        self.hidden_size = 128
        self.n_episodes = 15000
        self.model_path = os.path.join(os.path.dirname(__file__), "aiplane_model.pt")

        self.RUNWAY_LAT = 38.278404
        self.RUNWAY_LON = 27.161163
        self.RUNWAY_ELEVATION = 114.028
    
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
            
            if self.rl_enabled:
                epsilon = f"ε: {self.policy.epsilon:.3f}" if self.policy else "ε: N/A"
                xp.drawString(color, left + 5, y_pos, f"RL: READY | {epsilon}", 0, xp.Font_Basic)
                y_pos -= 15
            
            xp.drawString(color, left + 5, y_pos, "Left click: Reset | Right click: Start Training", 0, xp.Font_Basic)
            y_pos -= 15
            
            for message in self.debug_messages[-13:]:
                xp.drawString(color, left + 5, y_pos, message, 0, xp.Font_Basic)
                y_pos -= 12
                
        except Exception as e:
            pass
        
    def mouse_click_callback(self, inWindowID, x, y, inMouse, inRefcon):
        if inMouse == xp.MouseDown:
            self.start_training()

        return 1
        
    def key_callback(self, inWindowID, inKey, inFlags, inVirtualKey, inRefcon, losingFocus):
        pass
        
    def cursor_callback(self, inWindowID, x, y, inRefcon):
        return xp.CursorDefault
    
    def key_sniffer_callback(self, inChar, inFlags, inVirtualKey, inRefcon):
        if (inChar == 115 or inChar == 83) and (inFlags & xp.DownFlag):  # 's' or 'S' key down
            self.save_model()
            self.add_debug_message("Model saved via 'S' key!")
        return 0

    def onStart(self):
        xp.registerKeySniffer(self.key_sniffer_callback, 1, None)
        self.lat_ref = xp.findDataRef("sim/flightmodel/position/latitude")
        self.lon_ref = xp.findDataRef("sim/flightmodel/position/longitude")
        self.elevation_ref = xp.findDataRef("sim/flightmodel/position/elevation")
        self.pitch_ref = xp.findDataRef("sim/flightmodel/position/theta")
        self.bank_ref = xp.findDataRef("sim/flightmodel/position/phi")
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
        self.sim_speed_ref = xp.findDataRef("sim/time/sim_speed")
        xp.setDataf(self.sim_speed_ref, 8.0)
        
        self.create_display_window()
        self.initialize_rl_system()

    def after_physics(self):
        try:
            current_time = time.time()
            if current_time - self.last_clear_time >= 10.0:
                self.clear_debug_messages()
            self.step_training()
        except Exception as e:
            self.add_debug_message(f"Error: {e}")

    def print_current_state(self):
        try:
            from aiplane_models import feature_extractor
            
            self.debug_messages = []
            state = self.read_state()
            dummy_action = (0.0, 0.5, 0.0, 0.0)
            features = feature_extractor(state, dummy_action).squeeze().tolist()
            
            labels = [
                "Distance to runway (m)",
                "MSL (m)", 
                "Lateral deviation (m)",
                "Vertical deviation (m)",
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
            self.add_debug_message(f"RESETTING: {self.episode_count}")
            print(f"RESETTING: {self.episode_count}")
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
            self.set_actions(elevator=0.0, throttle=0.55, aileron=0.0, flaps=0.0)
            xp.setDataf(self.rudder_ref, 0.3)
            self.episode_count += 1
        except Exception as e:
            self.add_debug_message(f"RESET ERROR: {e}")

    def read_state(self):
        lat = xp.getDataf(self.lat_ref)
        lon = xp.getDataf(self.lon_ref)
        msl = xp.getDataf(self.elevation_ref)
        if msl < 0:
            msl = 0.0
        pitch = xp.getDataf(self.pitch_ref)
        bank = xp.getDataf(self.bank_ref)
        truepsi = xp.getDataf(self.truepsi_ref)

        vertical_speed = xp.getDataf(self.vertical_speed_ref)
        airspeed = xp.getDataf(self.airspeed_ref)
        distance = haversine_distance(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON)
        
        state = [
            distance,
            msl,
            lateral_deviation(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON, truepsi),  
            vertical_deviation(lat, lon, self.RUNWAY_LAT, self.RUNWAY_LON, msl, self.RUNWAY_ELEVATION),
            vertical_speed,
            pitch,
            bank,
            airspeed,
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
            
            input_size = 12
            self.model = AiplaneQNet(input_size, self.hidden_size)
            self.model.to(device)
            
            self.add_debug_message(f"Model path: {self.model_path}")
            self.add_debug_message(f"Path exists: {os.path.exists(self.model_path)}")
            
            if os.path.exists(self.model_path):
                try:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                    self.add_debug_message("Loaded existing model")
                except Exception as e:
                    self.add_debug_message(f"Could not load model: {e}")
            else:
                self.add_debug_message("No existing model found, starting fresh")
            
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
            self.max_episodes = 100000
            self.add_debug_message("Starting SARSA training...")
            self.training_state = None
            self.training_action = None
            
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
                
            if self.training_state is None:
                self.current_episode += 1
                self.training_state = self.environment.reset()
                self.training_action = self.policy.sample(self.training_state)
                if self.current_episode % 100 == 0:
                    self.add_debug_message(f"Episode {self.current_episode}/{self.max_episodes}")
                if self.current_episode % 10 == 0:
                    self.save_model()
                    self.add_debug_message(f"Auto-saved at episode {self.current_episode}")
                    
            new_state, reward = self.environment.step(self.training_state, self.training_action)
            new_action = self.policy.sample(new_state)
            
            elevator, throttle, aileron, flaps = self.training_action
            if self.environment.is_terminal(new_state):
                target_q = reward
            else:
                next_q = self.value_strategy.Q(new_state, new_action)
                target_q = reward + self.discount_rate * next_q
            
            self.value_strategy.update(self.training_state, self.training_action, target_q)
            
            self.training_state = new_state
            self.training_action = new_action
            if self.environment.is_terminal(new_state):
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
