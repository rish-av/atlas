import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel
import json
import re
from pathlib import Path
from tqdm import tqdm
import random
from dataembedder import CodeBERTEmbedder, extract_functions_regex, extract_global_code
from atlas_agent import FileLevelAgent, FunctionLevelAgent, LineLevelAgent, Critic


####################################
# Hierarchical RL Trainer with Teacher Forcing, Entropy & Critic
####################################

class HRLTrainer:
    def __init__(self, train_data, test_data, embedder, file_agent, function_agent, line_agent, 
                 file_optimizer, func_optimizer, line_optimizer, critic, critic_optimizer,
                 device="cpu", entropy_coef=0.01, reward_mode="sparse",
                 file_pretrain_epochs=1, func_pretrain_epochs=1, line_pretrain_epochs=1,
                 teacher_forcing_p_file=0.5, teacher_forcing_p_line=0.5, teacher_forcing_decay=0.95):
        self.train_data = train_data
        self.test_data = test_data
        self.embedder = embedder
        self.file_agent = file_agent.to(device)
        self.function_agent = function_agent.to(device)
        self.line_agent = line_agent.to(device)
        self.file_optimizer = file_optimizer
        self.func_optimizer = func_optimizer
        self.line_optimizer = line_optimizer
        self.critic = critic.to(device)
        self.critic_optimizer = critic_optimizer
        self.device = device
        self.entropy_coef = entropy_coef
        self.reward_mode = reward_mode
        self.file_pretrain_epochs = file_pretrain_epochs
        self.func_pretrain_epochs = func_pretrain_epochs
        self.line_pretrain_epochs = line_pretrain_epochs

        self.teacher_forcing_p_file = teacher_forcing_p_file
        self.teacher_forcing_p_line = teacher_forcing_p_line
        self.teacher_forcing_decay = teacher_forcing_decay
        self.max_num_funcs = 15
        self.k = 5

    def pretrain_file_agent(self):
        print("Pretraining File Agent ...")
        self.file_agent.train()
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(self.file_pretrain_epochs):
            epoch_loss = 0
            for sample in tqdm(self.train_data, desc=f"File Pretrain Epoch {epoch+1}"):
                bug_emb = torch.tensor(sample['stack_trace_embedding'], dtype=torch.float, device=self.device).unsqueeze(0)
                candidate_files = sample['file_contents']
                file_emb = self.embedder.get_file_embeddings(candidate_files).to(self.device).unsqueeze(0)
                file_probs, _ = self.file_agent(bug_emb, file_emb)
                target = torch.tensor([sample['correct_file_idx']], dtype=torch.long, device=self.device)
                loss = loss_fn(file_probs, target)
                epoch_loss += loss.item()
                self.file_optimizer.zero_grad()
                loss.backward()
                self.file_optimizer.step()
            print(f"File Pretrain Epoch {epoch+1}: Loss = {epoch_loss/len(self.train_data):.4f}")

    def pretrain_function_agent(self):
        print("Pretraining Function Agent ...")
        self.function_agent.train()
        loss_fn = nn.CrossEntropyLoss()
        count = 0
        for epoch in range(self.func_pretrain_epochs):
            epoch_loss = 0
            for sample in tqdm(self.train_data, desc=f"Function Pretrain Epoch {epoch+1}"):
                bug_emb = torch.tensor(sample['stack_trace_embedding'], dtype=torch.float, device=self.device).unsqueeze(0)
                candidate_files = sample['file_contents']
                file_emb = self.embedder.get_file_embeddings(candidate_files).to(self.device).unsqueeze(0)
                file_probs, _ = self.file_agent(bug_emb, file_emb)
                predicted_file_idx = torch.argmax(file_probs, dim=1).item()
                if predicted_file_idx != sample['correct_file_idx']:
                    continue
                selected_file_content = candidate_files[predicted_file_idx]
                candidate_functions, function_info = extract_functions_regex(selected_file_content)
                if len(candidate_functions) == 0:
                    global_code = extract_global_code(selected_file_content, [])
                    if global_code.strip() == "":
                        continue
                    candidate_functions = [global_code]
                    function_info = [{'name': "global", 'start_line': 1, 'end_line': len(global_code.split('\n'))}]
                correct_func_idx = -1
                for idx, info in enumerate(function_info):
                    if info['name'] == sample['buggy_function_name'] and info['start_line'] <= sample['buggy_line_number'] <= info['end_line']:
                        correct_func_idx = idx
                        break
                if correct_func_idx == -1:
                    for idx, info in enumerate(function_info):
                        if info['name'] == sample['buggy_function_name']:
                            correct_func_idx = idx
                            break
                if correct_func_idx == -1:
                    continue
                func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                func_probs, _ = self.function_agent(bug_emb, func_emb)
                target = torch.tensor([correct_func_idx], dtype=torch.long, device=self.device)
                loss = loss_fn(func_probs, target)
                epoch_loss += loss.item()
                count += 1
                self.func_optimizer.zero_grad()
                loss.backward()
                self.func_optimizer.step()
            if count > 0:
                print(f"Function Pretrain Epoch {epoch+1}: Loss = {epoch_loss/count:.4f} over {count} samples")
            else:
                print(f"Function Pretrain Epoch {epoch+1}: No valid samples")

    def pretrain_line_agent(self):
        print("Pretraining Line Agent ...")
        self.line_agent.train()
        loss_fn = nn.CrossEntropyLoss()
        count = 0
        for epoch in range(self.line_pretrain_epochs):
            epoch_loss = 0
            for sample in tqdm(self.train_data, desc=f"Line Pretrain Epoch {epoch+1}"):
                bug_emb = torch.tensor(sample['stack_trace_embedding'], dtype=torch.float, device=self.device).unsqueeze(0)
                candidate_files = sample['file_contents']
                file_emb = self.embedder.get_file_embeddings(candidate_files).to(self.device).unsqueeze(0)
                file_probs, _ = self.file_agent(bug_emb, file_emb)
                predicted_file_idx = torch.argmax(file_probs, dim=1).item()
                if predicted_file_idx != sample['correct_file_idx']:
                    continue
                selected_file_content = candidate_files[predicted_file_idx]
                candidate_functions, function_info = extract_functions_regex(selected_file_content)
                if len(candidate_functions) == 0:
                    continue
                correct_func_idx = -1
                for idx, info in enumerate(function_info):
                    if info['name'] == sample['buggy_function_name'] and info['start_line'] <= sample['buggy_line_number'] <= info['end_line']:
                        correct_func_idx = idx
                        break
                if correct_func_idx == -1:
                    for idx, info in enumerate(function_info):
                        if info['name'] == sample['buggy_function_name']:
                            correct_func_idx = idx
                            break
                if correct_func_idx == -1:
                    continue
                func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                func_probs, _ = self.function_agent(bug_emb, func_emb)
                predicted_func_idx = torch.argmax(func_probs, dim=1).item()
                if predicted_func_idx != correct_func_idx:
                    continue
                selected_function_content = candidate_functions[predicted_func_idx]
                selected_func_info = function_info[predicted_func_idx]
                candidate_lines = selected_function_content.split('\n')
                correct_line_idx = sample['buggy_line_number'] - selected_func_info['start_line']
                if not (0 <= correct_line_idx < len(candidate_lines)):
                    continue
                line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                line_probs, _ = self.line_agent(bug_emb, line_emb)
                target = torch.tensor([correct_line_idx], dtype=torch.long, device=self.device)
                loss = loss_fn(line_probs, target)
                epoch_loss += loss.item()
                count += 1
                self.line_optimizer.zero_grad()
                loss.backward()
                self.line_optimizer.step()
            if count > 0:
                print(f"Line Pretrain Epoch {epoch+1}: Loss = {epoch_loss/count:.4f} over {count} samples")
            else:
                print(f"Line Pretrain Epoch {epoch+1}: No valid samples")

    def sequential_pretrain(self):
        self.pretrain_file_agent()
        self.pretrain_function_agent()
        self.pretrain_line_agent()

    def train_episode(self, sample):
        bug_emb = torch.tensor(sample['stack_trace_embedding'], dtype=torch.float, device=self.device).unsqueeze(0)
        candidate_files = sample['file_contents']
        file_emb = self.embedder.get_file_embeddings(candidate_files).to(self.device).unsqueeze(0)
        file_probs, _ = self.file_agent(bug_emb, file_emb)
        file_entropy = -torch.sum(file_probs * torch.log(file_probs + 1e-8))
        if random.random() < self.teacher_forcing_p_file:
            file_action = sample['correct_file_idx']
            file_log_prob = torch.log(file_probs[0, sample['correct_file_idx']] + 1e-8)
        else:
            file_action = torch.argmax(file_probs, dim=1).item()
            file_log_prob = torch.log(file_probs[0, file_action] + 1e-8)
        file_correct = (file_action == sample['correct_file_idx'])
        candidate_functions, function_info = extract_functions_regex(candidate_files[file_action])
        if len(candidate_functions) == 0:
            global_code = extract_global_code(candidate_files[file_action], [])
            if global_code.strip() != "":
                candidate_functions = [global_code]
                function_info = [{'name': "global", 'start_line': 1, 'end_line': len(global_code.split('\n'))}]
        if candidate_functions is not None and len(candidate_functions) > self.max_num_funcs:
            buggy_candidate = None
            buggy_info = None
            for i, info in enumerate(function_info):
                if info['name'] == sample['buggy_function_name']:
                    buggy_candidate = candidate_functions[i]
                    buggy_info = function_info[i]
                    break
            if buggy_candidate is None:
                return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0
            else:
                non_buggy = [(func, info) for func, info in zip(candidate_functions, function_info)
                            if info['name'] != sample['buggy_function_name']]
                num_to_sample = min(self.max_num_funcs - 1, len(non_buggy))
                sampled = random.sample(non_buggy, num_to_sample)
                candidate_functions = [buggy_candidate] + [x[0] for x in sampled]
                function_info = [buggy_info] + [x[1] for x in sampled]
        
        func_action = -1
        func_log_prob = 0.0
        func_entropy = 0.0
        correct_func_idx = -1
        if file_correct and candidate_functions is not None and len(candidate_functions) > 0:
            func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
            func_probs, _ = self.function_agent(bug_emb, func_emb)
            func_entropy = -torch.sum(func_probs * torch.log(func_probs + 1e-8))
            func_action = torch.argmax(func_probs, dim=1).item()
            func_log_prob = torch.log(func_probs[0, func_action] + 1e-8)
            for idx, info in enumerate(function_info):
                if info['name'] == sample['buggy_function_name']:
                    correct_func_idx = idx
                    break
        function_correct = (file_correct and (func_action == correct_func_idx) and (correct_func_idx != -1))
        line_action = -1
        line_log_prob = 0.0
        line_entropy = 0.0
        correct_line_idx = None
        line_reward = 0.0
        if function_correct:
            selected_function_content = candidate_functions[func_action]
            selected_func_info = function_info[func_action]
            candidate_lines = selected_function_content.split('\n')
            correct_line_idx = sample['buggy_line_number'] - selected_func_info['start_line']
            if not (0 <= correct_line_idx < len(candidate_lines)):
                correct_line_idx = None
            else:
                line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                line_probs, _ = self.line_agent(bug_emb, line_emb)
                line_entropy = -torch.sum(line_probs * torch.log(line_probs + 1e-8))
                if random.random() < self.teacher_forcing_p_line:
                    line_action = correct_line_idx
                    line_log_prob = torch.log(line_probs[0, correct_line_idx] + 1e-8)
                else:
                    line_action = torch.argmax(line_probs, dim=1).item()
                    line_log_prob = torch.log(line_probs[0, line_action] + 1e-8)
                # Compute scaled reward if prediction is within k lines.
                error = abs(line_action - correct_line_idx)
                if error < self.k:
                    line_reward = 0.4 * (1 - error / self.k)
                else:
                    line_reward = 0.0
        if file_correct and not function_correct:
            total_reward = 0.3
        elif file_correct and function_correct:
            total_reward = 0.3 + 2.0 + line_reward*20  # Total ranges from 0.6 (if line is far off) to 1.0 (if exact).
        else:
            total_reward = 0.0
        critic_value = self.critic(bug_emb)
        advantage = total_reward - critic_value.item()
        total_log_prob = file_log_prob + func_log_prob + line_log_prob
        actor_loss = - total_log_prob * advantage - self.entropy_coef * (file_entropy + func_entropy + line_entropy)
        critic_loss = 0.5 * advantage ** 2
        loss = actor_loss + critic_loss

        # print (f'file action = {file_action}, function action = {func_action}, line action = {line_action}')ß
        # print(f'file reward = {file_correct}, function reward = {function_correct}, line reward = {line_reward}, total reward = {total_reward}')
        return loss, total_reward


    def compute_metrics(self):
        total_samples = len(self.test_data)
        file_correct_count = 0
        function_correct_count = 0
        line_accuracy_total = 0.0  # Accumulate scaled line accuracy.
        overall_correct_count = 0

        for sample in self.test_data:
            bug_emb = torch.tensor(sample['stack_trace_embedding'], dtype=torch.float, device=self.device).unsqueeze(0)
            candidate_files = sample['file_contents']
            file_emb = self.embedder.get_file_embeddings(candidate_files).to(self.device).unsqueeze(0)
            file_probs, _ = self.file_agent(bug_emb, file_emb)
            file_pred = torch.argmax(file_probs, dim=1).item()
            file_correct = (file_pred == sample['correct_file_idx'])
            if file_correct:
                file_correct_count += 1
                selected_file_content = candidate_files[file_pred]
                candidate_functions, function_info = extract_functions_regex(selected_file_content)
                if len(candidate_functions) == 0:
                    global_code = extract_global_code(selected_file_content, [])
                    if global_code.strip() != "":
                        candidate_functions = [global_code]
                        function_info = [{'name': "global", 'start_line': 1, 'end_line': len(global_code.split('\n'))}]
                if candidate_functions is None or len(candidate_functions) == 0:
                    continue
                func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                func_probs, _ = self.function_agent(bug_emb, func_emb)
                func_pred = torch.argmax(func_probs, dim=1).item()
                correct_func_idx = -1
                for idx, info in enumerate(function_info):
                    if info['name'] == sample['buggy_function_name']:
                        correct_func_idx = idx
                        break
                function_correct = (func_pred == correct_func_idx) and (correct_func_idx != -1)
                if function_correct:
                    function_correct_count += 1
                    selected_function_content = candidate_functions[func_pred]
                    selected_func_info = function_info[func_pred]
                    candidate_lines = selected_function_content.split('\n')
                    correct_line_idx = sample['buggy_line_number'] - selected_func_info['start_line']
                    if not (0 <= correct_line_idx < len(candidate_lines)):
                        line_acc = 0.0
                    else:
                        line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                        line_probs, _ = self.line_agent(bug_emb, line_emb)
                        line_pred = torch.argmax(line_probs, dim=1).item()
                        error = abs(line_pred - correct_line_idx)
                        if error < self.k:
                            line_acc = 1 - error / self.k
                        else:
                            line_acc = 0.0
                        # Count overall as correct if error == 0.
                        if error == 0:
                            overall_correct_count += 1
                    line_accuracy_total += line_acc
        metrics = {
            'file_accuracy': file_correct_count / total_samples,
            'function_accuracy': function_correct_count / total_samples,
            'line_accuracy': line_accuracy_total / total_samples,
            'overall_accuracy': overall_correct_count / total_samples
        }
        return metrics



    def train(self, epochs):
        print("Starting Sequential Pretraining ...")
        self.sequential_pretrain()
        print("Pretraining Completed. Starting Joint Training...")
        self.file_agent.train()
        self.function_agent.train()
        self.line_agent.train()
        self.critic.train()
        total_rewards = 0
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_reward = 0
            for sample in self.train_data:
                loss, reward = self.train_episode(sample)
                epoch_loss += loss.item()
                epoch_reward += reward
                self.file_optimizer.zero_grad()
                self.func_optimizer.zero_grad()
                self.line_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.file_optimizer.step()
                self.func_optimizer.step()
                self.line_optimizer.step()
                self.critic_optimizer.step()
            avg_loss = epoch_loss / len(self.train_data)
            avg_reward = epoch_reward / len(self.train_data)
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
            print("Evaluation Metrics:", self.compute_metrics())
            total_rewards += epoch_reward
            # Decay teacher forcing probabilities.
            self.teacher_forcing_p_file *= self.teacher_forcing_decay
            self.teacher_forcing_p_line *= self.teacher_forcing_decay
            print(f"Teacher Forcing - File: {self.teacher_forcing_p_file:.4f}, Line: {self.teacher_forcing_p_line:.4f}")
        print(f"Total Avg Reward over {epochs} epochs: {total_rewards/epochs:.4f}")

####################################
# Main Training Execution
####################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset (JSON file with samples having keys:
    # ['stack_trace_embedding', 'file_paths', 'file_contents', 'correct_file_idx', 'buggy_line_number', 'buggy_function_name'])
    dataset_path = Path("/home/ubuntu/atlas/data/simplified_dataset/train.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Split dataset into train and test.
    split_idx = int(0.9 * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    # Instantiate embedder and agents.
    embedder = CodeBERTEmbedder(device=device)
    file_agent = FileLevelAgent(bug_emb_dim=768, file_emb_dim=768, lstm_hidden_dim=128,
                                mlp_hidden_dim=128, mlp_out_dim=128,
                                use_projection=False, use_hierarchical_pool=False).to(device)
    func_agent = FunctionLevelAgent(bug_emb_dim=768, func_emb_dim=768,
                                    mlp_hidden_dim=128, mlp_out_dim=128,
                                    use_projection=False, use_hierarchical_pool=False).to(device)
    line_agent = LineLevelAgent(bug_emb_dim=768, line_emb_dim=768, lstm_hidden_dim=128,
                                mlp_hidden_dim=128, mlp_out_dim=128,
                                use_projection=False, use_hierarchical_pool=False).to(device)
    
    # Instantiate critic.
    critic = Critic(bug_emb_dim=768, hidden_dim=128).to(device)
    
    # Create optimizers.
    file_optimizer = optim.Adam(file_agent.parameters(), lr=1e-3)
    func_optimizer = optim.Adam(func_agent.parameters(), lr=1e-3)
    line_optimizer = optim.Adam(line_agent.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    # Instantiate HRLTrainer with teacher forcing, entropy, and critic.
    trainer = HRLTrainer(
        train_data=train_data,
        test_data=test_data,
        embedder=embedder,
        file_agent=file_agent,
        function_agent=func_agent,
        line_agent=line_agent,
        file_optimizer=file_optimizer,
        func_optimizer=func_optimizer,
        line_optimizer=line_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        device=device,
        entropy_coef=0.01,
        reward_mode="sparse",
        file_pretrain_epochs=5,
        func_pretrain_epochs=5,
        line_pretrain_epochs=5,
        teacher_forcing_p_file=0.5,
        teacher_forcing_p_line=0.5,
        teacher_forcing_decay=0.95
    )
    
    # Train the model.
    trainer.train(epochs=10)
