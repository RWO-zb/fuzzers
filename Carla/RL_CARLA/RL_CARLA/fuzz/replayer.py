import numpy as np
from scipy.stats import multivariate_normal
import copy
import tqdm
import carla

class replayer:
    def __init__(self):
        self.corpus = []
        self.rewards = []
        self.entropy = []
        self.coverage = []
        self.original = []
        self.envsetting = []
        self.state_cvg = []

        self.sequences = []
        self.current_pose = None
        self.current_reward = None
        self.current_entropy = None
        self.current_coverage = None
        self.current_index = None
        self.current_nvsetting = None
        self.replay_list = None
        self.current_vehicle_info = None

    def get_pose(self):
        if self.replay_list == None:
            choose_index = 0
        else:
            choose_index = self.replay_list[-1]
            self.replay_list.pop(len(self.replay_list)-1)

        self.current_index = choose_index
        self.current_pose = self.corpus[choose_index][0]
        self.current_vehicle_info = self.corpus[choose_index][1]
        self.current_reward = self.rewards[choose_index]
        self.current_entropy = self.entropy[choose_index]
        self.current_coverage = self.coverage[choose_index]
        self.current_envsetting = self.envsetting[choose_index]

        self.corpus.pop(choose_index)
        self.rewards.pop(choose_index)
        self.entropy.pop(choose_index)
        self.coverage.pop(choose_index)
        self.envsetting.pop(choose_index)
        self.current_index = None

        return self.current_pose

    def get_vehicle_info(self):
        return self.current_vehicle_info

    def store(self, current_pose, rewards, entropy, cvg, original, further_envsetting):
        pose = current_pose[0]
        newpose = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
        vehicle_info = current_pose[1]
        new_vehicle_info = []
        for i in range(len(vehicle_info)):
            pose = vehicle_info[i][1]
            v_1 = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
            temp = (vehicle_info[i][0], v_1, vehicle_info[i][2], vehicle_info[i][3])
            new_vehicle_info.append(temp)

        self.corpus.append((newpose, new_vehicle_info))
        self.rewards.append(rewards)
        self.entropy.append(entropy)
        self.coverage.append(cvg)
        self.original.append(original)
        self.envsetting.append(further_envsetting)

    def drop_current(self):
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            self.envsetting.pop(choose_index)
            self.current_index = None

    def __getstate__(self):
        state = self.__dict__.copy()

        def serialize_data(data):
            if isinstance(data, list):
                return [serialize_data(item) for item in data]
            elif isinstance(data, tuple):
                return tuple(serialize_data(item) for item in data)
            elif isinstance(data, carla.Transform):
                return {
                    '__carla_type__': 'Transform',
                    'x': data.location.x, 'y': data.location.y, 'z': data.location.z,
                    'pitch': data.rotation.pitch, 'yaw': data.rotation.yaw, 'roll': data.rotation.roll
                }
            elif isinstance(data, carla.Location):
                return {
                    '__carla_type__': 'Location',
                    'x': data.x, 'y': data.y, 'z': data.z
                }
            elif isinstance(data, carla.Rotation):
                return {
                    '__carla_type__': 'Rotation',
                    'pitch': data.pitch, 'yaw': data.yaw, 'roll': data.roll
                }
            return data
        keys_to_process = ['corpus', 'original', 'current_pose', 'current_vehicle_info']
        for key in keys_to_process:
            if key in state and state[key] is not None:
                state[key] = serialize_data(state[key])
        
        return state

    def __setstate__(self, state):
        def deserialize_data(data):
            if isinstance(data, list):
                return [deserialize_data(item) for item in data]
            elif isinstance(data, tuple):
                return tuple(deserialize_data(item) for item in data)
            elif isinstance(data, dict) and '__carla_type__' in data:
                if data['__carla_type__'] == 'Transform':
                    return carla.Transform(
                        carla.Location(x=data['x'], y=data['y'], z=data['z']),
                        carla.Rotation(pitch=data['pitch'], yaw=data['yaw'], roll=data['roll'])
                    )
                elif data['__carla_type__'] == 'Location':
                    return carla.Location(x=data['x'], y=data['y'], z=data['z'])
                elif data['__carla_type__'] == 'Rotation':
                    return carla.Rotation(pitch=data['pitch'], yaw=data['yaw'], roll=data['roll'])
            return data
        for key, value in state.items():
            state[key] = deserialize_data(value)
        
        self.__dict__.update(state)