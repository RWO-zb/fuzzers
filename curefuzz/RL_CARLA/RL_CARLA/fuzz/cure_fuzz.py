import numpy as np
import copy
import carla
import torch.nn as nn
import torch
import torch.optim as optim
import math

class RND(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=16):
        super(RND, self).__init__()
        self.target_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.predictor_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # Initialize the target network with random weights
        for m in self.target_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
        # Make target network's parameters not trainable
        for param in self.target_net.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        target_out = self.target_net(x)
        predictor_out = self.predictor_net(x)
        return target_out, predictor_out


class cure:
    def __init__(self, input_size=7, hidden_size=64, output_size=16, learning_rate=1e-6):
        self.corpus = []
        self.final_state = []
        self.rewards = []
        self.result = []
        self.entropy = []
        self.intrinsic_reward = []
        self.original = []
        self.count = []
        self.envsetting = []

        self.sequences = []
        self.current_pose = None
        self.current_final_state = None
        self.current_reward = None
        self.current_entropy = None
        self.current_intrinsic_reward = None
        self.current_original = None
        self.current_index = None
        self.current_envsetting = None
        self.current_vehicle_info = None
        
        self.rnd = RND(input_size, hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnd = self.rnd.to(self.device)
        self.optimizer = optim.Adam(list(self.rnd.predictor_net.parameters()), lr=learning_rate)

    def get_pose(self, alpha=0.01, beta=0.001, gamma=1):
        if not self.corpus:
            return None

        new_prob = []
        for i in range(len(self.corpus)):
            # [修改 1] 恢复老代码的计算逻辑，移除 try-except 和 1e100 溢出保护
            # 这允许 math.exp 产生较大的值，避免因人为截断导致的种子选择死锁
            prob = (math.exp(-self.rewards[i]*alpha) + math.exp(self.intrinsic_reward[i]*beta) + self.entropy[i]*gamma)
            new_prob.append(prob)
        
        new_prob = np.array(new_prob)
        
        # [修改 2] 保留新代码的负概率截断和清洗逻辑 (User Request)
        # 确保概率非负且无非法值，防止 np.random.choice 崩溃
        new_prob = np.nan_to_num(new_prob, nan=0.0, posinf=None, neginf=0.0)
        new_prob = np.maximum(new_prob, 1e-6)
        
        prob_sum = new_prob.sum()
        if prob_sum <= 0:
            final_probs = np.ones(len(self.corpus)) / len(self.corpus)
        else:
            final_probs = new_prob / prob_sum
                    
        choose_index = np.random.choice(range(len(self.corpus)), 1, p=final_probs)[0]
        
        self.count[choose_index] -= 1
        self.current_index = choose_index
        self.current_pose = self.corpus[choose_index][0]
        self.current_vehicle_info = self.corpus[choose_index][1]
        self.current_final_state = self.final_state[choose_index]
        self.current_reward = self.rewards[choose_index]
        self.current_entropy = self.entropy[choose_index]
        self.current_intrinsic_reward = self.intrinsic_reward[choose_index]
        self.current_original = self.original[choose_index]
        self.current_envsetting = self.envsetting[choose_index]
        
        if self.count[choose_index] <= 0:
            self.drop_current()
            
        return self.current_pose

    def get_vehicle_info(self):
        return self.current_vehicle_info

    def add_crash(self, result_pose):
        self.result.append(result_pose)
        self.drop_current()

    def further_mutation(self, current_pose, rewards, entropy, intrinsic_reward, final_state, original, further_envsetting):
        choose_index = self.current_index
        pose = current_pose[0]
        newpose = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
        vehicle_info = current_pose[1]
        new_vehicle_info = []
        if vehicle_info:
            for i in range(len(vehicle_info)):
                pose = vehicle_info[i][1]
                v_1 = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
                temp = (vehicle_info[i][0], v_1, vehicle_info[i][2], vehicle_info[i][3])
                new_vehicle_info.append(temp)
        copy_pose = (newpose, new_vehicle_info)
        copy_envsetting = list(further_envsetting)

        if choose_index is not None and choose_index < len(self.corpus):
            self.corpus[choose_index] = copy_pose
            self.final_state[choose_index] = final_state
            self.rewards[choose_index] = rewards
            self.entropy[choose_index] = entropy
            self.intrinsic_reward[choose_index] = intrinsic_reward
            self.count[choose_index] = 5
            self.envsetting[choose_index] = copy_envsetting
        else:
            self.corpus.append(copy_pose)
            self.final_state.append(final_state)
            self.rewards.append(rewards)
            self.entropy.append(entropy)
            self.intrinsic_reward.append(intrinsic_reward)
            self.original.append(original)
            self.count.append(5)
            self.envsetting.append(copy_envsetting)
    
    def mutation(self, pose):
        newpose = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
        newpose.location.x = pose.location.x + np.random.uniform(-0.15, 0.15)
        newpose.location.y = pose.location.y + np.random.uniform(-0.15, 0.15)
        newpose.rotation.yaw = pose.rotation.yaw + np.random.uniform(-5, 5)
        self.current_pose = newpose
        return newpose
    
    def vehicle_mutate(self, vehicle_info):
        new_vehicle_info = []
        if vehicle_info is None:
            return []
        for i in range(len(vehicle_info)):
            pose = vehicle_info[i][1]
            v_1 = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
            v_1.location.x += np.random.uniform(-0.1, 0.1)
            v_1.location.y += np.random.uniform(-0.1, 0.1)
            temp = (vehicle_info[i][0], v_1, vehicle_info[i][2], vehicle_info[i][3])
            new_vehicle_info.append(temp)

        self.current_vehicle_info = new_vehicle_info
        return self.current_vehicle_info
    
    def drop_current(self):
        choose_index = self.current_index
        if self.current_index is not None and choose_index < len(self.corpus):
            self.corpus.pop(choose_index)
            self.final_state.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.intrinsic_reward.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.envsetting.pop(choose_index)
            self.current_index = None

    def flatten_states(self, states):
        states = np.array(states)
        states_cond = np.zeros((states.shape[0]-1, states.shape[1] * 2))
        for i in range(states.shape[0]-1):
            states_cond[i] = np.hstack((states[i], states[i + 1]))
        return states, states_cond
    
    def compute_intrinsic_reward(self, states):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states).to(self.device)
            target_out, predictor_out = self.rnd(state_tensor)
            intrinsic_reward = torch.pow(target_out - predictor_out, 2).sum()
        intrinsic_reward = intrinsic_reward.cpu().numpy()
        return intrinsic_reward
    
    def train_rnd(self, states, intrinsic_reward_scale=1e-6, l2_reg_coeff=1e-6):
        state_tensor = torch.FloatTensor(states).to(self.device)
        target_out, predictor_out = self.rnd(state_tensor)
        
        intrinsic_reward = torch.pow(target_out - predictor_out, 2).sum(dim=1, keepdim=True) * intrinsic_reward_scale
        loss = torch.mean(intrinsic_reward)
        l2_reg = 0
        for param in self.rnd.predictor_net.parameters():
            l2_reg += torch.norm(param)
        loss = loss + l2_reg_coeff * l2_reg
        
        # [修改 3] 严格对齐老代码：注释掉/移除 optimizer.zero_grad()
        # 允许梯度累积，使网络产生漂移，维持较高的内在奖励（好奇心），
        # 这是老代码能发现更多 Crash 的关键机制。
        if not torch.isnan(loss):
            # self.optimizer.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rnd.predictor_net.parameters(), 5)
            self.optimizer.step()
    
        return loss.item()