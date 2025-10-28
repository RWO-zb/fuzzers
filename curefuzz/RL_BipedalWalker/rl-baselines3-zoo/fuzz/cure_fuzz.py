import numpy as np
import copy
import torch.nn as nn
import torch
import torch.optim as optim
import math

class RND(nn.Module):
    def __init__(self, input_size=24, hidden_size=256, output_size=256):
        super(RND, self).__init__()
        self.target_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.predictor_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
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
            
class CureFuzz:
    def __init__(self, input_size=24, hidden_size=64, output_size=16,learning_rate=1e-4):
        self.corpus = []
        self.final_state = []
        self.rewards = []
        self.result = []
        self.entropy = []
        self.intrinsic_reward = []
        self.original = []
        self.count = []
        self.current_pose = None
        self.current_final_state = None
        self.current_reward = None
        self.current_entropy = None
        self.current_intrinsic_reward = None
        self.current_original = None
        self.current_index = None

         # --- 新增属性 ---
        self.next_seed_id = 0       # 用于分配唯一ID的计数器
        self.seed_ids = []          # 列表：存储每个种子的唯一ID
        self.parent_ids = []        # 列表：存储每个种子的父种子ID
        self.selection_counts = []  # 列表：存储每个种子被挑选的次数
        self.current_parent_id = None # 临时变量：存储当前被选中种子的ID
        # --- 结束新增 ---

        self.rnd = RND(input_size, hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnd = self.rnd.to(self.device)
        self.optimizer = optim.Adam(list(self.rnd.predictor_net.parameters()), lr=learning_rate)



    def get_pose(self):
        new_prob = []
        for i in range(len(self.corpus)):
            prob = math.exp(-self.rewards[i]*0.1)+self.entropy[i]
            new_prob.append(prob)
        choose_index = np.random.choice(range(len(self.corpus)), 1, p=new_prob/np.array(new_prob).sum())[0]

        # --- 新增：更新挑选次数和设置当前父ID ---
        self.selection_counts[choose_index] += 1
        self.current_parent_id = self.seed_ids[choose_index]
        # --- 结束新增 ---

        # --- 新增：捕获要返回的种子信息 ---
        selected_info = {
            'seed_id': self.seed_ids[choose_index],
            'seed_state': self.corpus[choose_index],
            'parent_id': self.parent_ids[choose_index],
            'selection_count': self.selection_counts[choose_index]
        }
        # --- 结束新增 ---

        self.count[choose_index] -= 1
        self.current_index = choose_index
        self.current_pose = self.corpus[choose_index]
        self.current_final_state = self.final_state[choose_index]
        self.current_reward = self.rewards[choose_index]
        self.current_entropy = self.entropy[choose_index]
        self.current_intrinsic_reward = self.intrinsic_reward[choose_index]
        self.current_original = self.original[choose_index]
        if self.count[choose_index] <= 0:
            self.corpus.pop(choose_index)
            self.final_state.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.intrinsic_reward.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)

            # --- 新增：同步删除新属性 ---
            self.seed_ids.pop(choose_index)
            self.parent_ids.pop(choose_index)
            self.selection_counts.pop(choose_index)
            # --- 结束新增 ---

            self.current_index = None

            self.current_parent_id = None# 因为种子被移除了
        return  selected_info


    def add_crash(self, result_pose):
        self.result.append(result_pose)
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.final_state.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.intrinsic_reward.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)

            # --- 新增：同步删除新属性 ---
            self.seed_ids.pop(choose_index)
            self.parent_ids.pop(choose_index)
            self.selection_counts.pop(choose_index)
            # --- 结束新增 ---
            self.current_index = None

            self.current_parent_id = None# 因为种子被移除了
    

    def further_mutation(self, current_pose, rewards, entropy, intrinsic_reward, final_state, original):
        choose_index = self.current_index
        copy_pose = copy.deepcopy(current_pose)

        # --- 获取新的唯一ID ---
        new_id = self.next_seed_id
        self.next_seed_id += 1
        # --- 结束 ---

        if choose_index != None:
            self.corpus[choose_index] = copy_pose
            self.final_state[choose_index] = final_state
            self.rewards[choose_index] = rewards
            self.entropy[choose_index] = entropy
            self.intrinsic_reward[choose_index] = intrinsic_reward
            self.count[choose_index] = 5
            # --- 新增：更新新属性（替换） ---
            self.seed_ids[choose_index] = new_id         # 替换为新的ID
            self.parent_ids[choose_index] = self.current_parent_id # 记录父ID
            self.selection_counts[choose_index] = 0    # 重置挑选次数
            # --- 结束新增 ---
        else:
            self.corpus.append(copy_pose)
            self.final_state.append(final_state)
            self.rewards.append(rewards)
            self.entropy.append(entropy)
            self.intrinsic_reward.append(intrinsic_reward)
            self.original.append(original)
            self.count.append(5)
            # --- 新增：添加新属性（追加） ---
            self.seed_ids.append(new_id)
            self.parent_ids.append(None) # 初始种子没有父ID
            self.selection_counts.append(0) # 初始挑选次数为0
            # --- 结束新增 ---


    def mutation(self, states):
        delta_states = np.random.choice(2, 15, p=[0.9, 0.1])
        # if np.sum(delta_states) == 0:
        #     delta_states[0] = 1
        mutate_states = states + delta_states
        mutate_states = np.remainder(mutate_states, 4)
        mutate_states = np.clip(mutate_states, 1, 3)
        
        return mutate_states

    def drop_current(self):
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.final_state.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.intrinsic_reward.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            # --- 新增：同步删除新属性 ---
            self.seed_ids.pop(choose_index)
            self.parent_ids.pop(choose_index)
            self.selection_counts.pop(choose_index)
            # --- 结束新增 ---
            self.current_index = None
            self.current_parent_id = None# 因为种子被移除了

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
    
    
    def train_rnd(self, states, intrinsic_reward_scale=1.0, l2_reg_coeff=1e-4):
        state_tensor = torch.FloatTensor(states).to(self.device)
        target_out, predictor_out = self.rnd(state_tensor)
        intrinsic_reward = torch.pow(target_out[0,:] - predictor_out[0,:] , 2).sum(dim=0, keepdim=True)                
        mse_loss = nn.MSELoss()(predictor_out, target_out)

        l2_reg = 0
        for param in self.rnd.predictor_net.parameters():
            l2_reg += torch.norm(param)
        loss = mse_loss + l2_reg_coeff * l2_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return intrinsic_reward.cpu().detach().numpy()
