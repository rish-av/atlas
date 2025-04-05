import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Modified HRLTrainer using Precomputed Embeddings from JSON ---

class HRLTrainer:
    def __init__(self, train_data, test_data, file_agent, function_agent, line_agent, 
                 file_optimizer, func_optimizer, line_optimizer, device="cpu", 
                 entropy_coef=0.01, reward_mode="sparse", alpha=0.5, tau=0.1,
                 file_pretrain_epochs=0, func_pretrain_epochs=0, line_pretrain_epochs=0,
                 line_tolerance=2):
        """
        reward_mode: "sparse", "intermediate", "ranking", or "mixed"
        alpha: weight for intermediate reward in mixed mode (0<=alpha<=1)
        tau: temperature parameter for ranking reward approximations
        Pretrain epochs: number of epochs to pretrain each agent individually.
        line_tolerance: tolerance threshold for line prediction (if prediction is within k of target)
        """
        self.dataset = train_data
        self.test_dataset = test_data
        self.file_agent = file_agent.to(device)
        self.function_agent = function_agent.to(device)
        self.line_agent = line_agent.to(device)
        self.file_optimizer = file_optimizer
        self.func_optimizer = func_optimizer
        self.line_optimizer = line_optimizer
        self.device = device
        self.entropy_coef = entropy_coef
        self.reward_mode = reward_mode
        self.alpha = alpha
        self.tau = tau
        self.file_pretrain_epochs = file_pretrain_epochs
        self.func_pretrain_epochs = func_pretrain_epochs
        self.line_pretrain_epochs = line_pretrain_epochs
        self.line_tolerance = line_tolerance

    def pretrain_file_agent(self):
        self.file_agent.train()
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(self.file_pretrain_epochs):
            epoch_loss = 0
            for sample in self.dataset:
                bug_emb = sample['stack_trace_embedding'].unsqueeze(0).to(self.device)
                file_emb = sample['file_embeddings'].unsqueeze(0).to(self.device)
                file_probs, _ = self.file_agent(bug_emb, file_emb)
                if len(sample['file_embeddings']) <= sample['correct_file_idx']:
                    target = F.one_hot(torch.tensor(sample['correct_file_idx']-1, dtype=torch.long, device=self.device), 
                                       num_classes=len(sample['file_embeddings'])).float()
                else:
                    target = F.one_hot(torch.tensor(sample['correct_file_idx'], dtype=torch.long, device=self.device), 
                                       num_classes=len(sample['file_embeddings'])).float()
                loss = loss_fn(file_probs, target.unsqueeze(0))
                epoch_loss += loss.item()
                self.file_optimizer.zero_grad()
                loss.backward()
                self.file_optimizer.step()
            print(f"Pretrain File Agent Epoch {epoch+1}: Loss = {epoch_loss/len(self.dataset):.4f}")

    def pretrain_function_agent(self):
        self.function_agent.train()
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(self.func_pretrain_epochs):
            epoch_loss = 0
            for sample in self.dataset:
                bug_emb = sample['stack_trace_embedding'].unsqueeze(0).to(self.device)
                func_emb = sample['function_embeddings'].unsqueeze(0).to(self.device)
                func_probs, _ = self.function_agent(bug_emb, func_emb)
                if sample['correct_function_idx'] == -1:
                    continue
                target = F.one_hot(torch.tensor(sample['correct_function_idx'], dtype=torch.long, device=self.device), 
                                   num_classes=len(sample['function_embeddings'])).float().unsqueeze(0)
                loss = loss_fn(func_probs, target)
                epoch_loss += loss.item()
                self.func_optimizer.zero_grad()
                loss.backward()
                self.func_optimizer.step()
            print(f"Pretrain Function Agent Epoch {epoch+1}: Loss = {epoch_loss/len(self.dataset):.4f}")

    def pretrain_line_agent(self):
        self.line_agent.train()
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(self.line_pretrain_epochs):
            epoch_loss = 0
            for sample in self.dataset:
                bug_emb = sample['stack_trace_embedding'].unsqueeze(0).to(self.device)
                line_emb = sample['line_embeddings'].unsqueeze(0).to(self.device)
                line_probs, _ = self.line_agent(bug_emb, line_emb)
                # If correct_line_idx is empty, skip sample
                if len(sample['correct_line_idx']) == 0:
                    continue
                # Use the first correct index for pretraining
                target_idx = sample['correct_line_idx'][0] if isinstance(sample['correct_line_idx'], list) else sample['correct_line_idx']
                target = F.one_hot(torch.tensor(target_idx, dtype=torch.long, device=self.device), 
                                   num_classes=len(sample['line_embeddings'])).float()
                loss = loss_fn(line_probs, target)
                epoch_loss += loss.item()
                self.line_optimizer.zero_grad()
                loss.backward()
                self.line_optimizer.step()
            print(f"Pretrain Line Agent Epoch {epoch+1}: Loss = {epoch_loss/len(self.dataset):.4f}")

    def pretrain_all(self):
        if self.file_pretrain_epochs > 0:
            print("Pretraining File Agent")
            self.pretrain_file_agent()
        if self.func_pretrain_epochs > 0:
            print("Pretraining Function Agent")
            self.pretrain_function_agent()
        if self.line_pretrain_epochs > 0:
            print("Pretraining Line Agent")
            self.pretrain_line_agent()

    def _compute_intermediate_reward_file(self, file_probs, file_scores, correct_idx):
        return file_probs[0, correct_idx]

    def _compute_ranking_reward_file(self, file_scores, correct_idx):
        score_correct = file_scores[0, correct_idx]
        diff = file_scores[0] - score_correct
        mask = torch.ones_like(diff, dtype=torch.bool)
        mask[correct_idx] = False
        rank_approx = 1 + torch.sum(torch.sigmoid(diff[mask] / self.tau))
        return 1 / torch.log2(rank_approx + 1)

    def _compute_intermediate_reward_func(self, func_probs, func_scores, correct_idx):
        return func_probs[0, correct_idx]

    def _compute_ranking_reward_func(self, func_scores, correct_idx):
        score_correct = func_scores[0, correct_idx]
        diff = func_scores[0] - score_correct
        mask = torch.ones_like(diff, dtype=torch.bool)
        mask[correct_idx] = False
        rank_approx = 1 + torch.sum(torch.sigmoid(diff[mask] / self.tau))
        return 1 / torch.log2(rank_approx + 1)

    def _compute_reward_line(self, line_probs, target_idx):
        return line_probs[0, target_idx]

    def train_episode(self, sample):
        bug_emb = sample["stack_trace_embedding"].unsqueeze(0).to(self.device)
        file_emb = sample["file_embeddings"].unsqueeze(0).to(self.device)
        func_emb = sample["function_embeddings"].unsqueeze(0).to(self.device)
        line_emb = sample["line_embeddings"].unsqueeze(0).to(self.device)
        
        file_probs, file_scores = self.file_agent(bug_emb, file_emb)
        m_file = torch.distributions.Categorical(file_probs)
        file_action = m_file.sample()
        log_prob_file = m_file.log_prob(file_action)
        
        func_probs, func_scores = self.function_agent(bug_emb, func_emb)
        m_func = torch.distributions.Categorical(func_probs)
        func_action = m_func.sample()
        log_prob_func = m_func.log_prob(func_action)
        
        line_probs, line_scores = self.line_agent(bug_emb, line_emb)
        m_line = torch.distributions.Categorical(line_probs)
        line_action = m_line.sample()
        log_prob_line = m_line.log_prob(line_action)
        
        if self.reward_mode == "sparse":
            file_correct = (file_action.item() == sample['correct_file_idx'])
            func_correct = (func_action.item() == sample['correct_function_idx'])
            # Check if predicted line is within tolerance of any correct line index
            line_correct = any(abs(line_action.item() - idx) <= self.line_tolerance for idx in sample['correct_line_idx'])
            reward = 1.0 if file_correct and func_correct and line_correct else 0.0
        elif self.reward_mode in ["intermediate", "ranking", "mixed"]:
            r_file_int = self._compute_intermediate_reward_file(file_probs, file_scores, sample['correct_file_idx'])
            r_func_int = self._compute_intermediate_reward_func(func_probs, func_scores, sample['correct_function_idx'])
            # For reward calculation, use the first correct line index if multiple are provided
            target_line_idx = sample['correct_line_idx'][0] if isinstance(sample['correct_line_idx'], list) else sample['correct_line_idx']
            r_line_int = self._compute_reward_line(line_probs, target_line_idx)
            r_file_rank = self._compute_ranking_reward_file(file_scores, sample['correct_file_idx'])
            r_func_rank = self._compute_ranking_reward_func(func_scores, sample['correct_function_idx'])
            r_line_rank = r_line_int

            if self.reward_mode == "intermediate":
                reward = (r_file_int + r_func_int + r_line_int) / 3.0
            elif self.reward_mode == "ranking":
                reward = (0.1 * r_file_rank + 0.2 * r_func_rank + r_line_rank) / 3.0
            elif self.reward_mode == "mixed":
                reward_int = (r_file_int + r_func_int + r_line_int) / 3.0
                reward_rank = (r_file_rank + r_func_rank + r_line_rank) / 3.0
                reward = self.alpha * reward_int + (1 - self.alpha) * reward_rank
        else:
            reward = 0.0

        total_log_prob = log_prob_file + log_prob_func + log_prob_line
        entropy = m_file.entropy() + m_func.entropy() + m_line.entropy()
        loss = - total_log_prob * reward - self.entropy_coef * entropy
        return loss, reward

    def compute_metrics(self):
        total_samples = len(self.test_dataset)
        file_correct_count = 0
        function_correct_count = 0
        line_correct_count = 0
        overall_correct_count = 0
        for sample in self.test_dataset:
            bug_emb = sample["stack_trace_embedding"].unsqueeze(0).to(self.device)
            file_emb = sample["file_embeddings"].unsqueeze(0).to(self.device)
            func_emb = sample["function_embeddings"].unsqueeze(0).to(self.device)
            line_emb = sample["line_embeddings"].unsqueeze(0).to(self.device)
            
            file_probs, _ = self.file_agent(bug_emb, file_emb)
            file_pred = torch.argmax(file_probs, dim=1).item()
            file_correct = (file_pred == sample['correct_file_idx'])
            if file_correct:
                file_correct_count += 1
                
            func_probs, _ = self.function_agent(bug_emb, func_emb)
            func_pred = torch.argmax(func_probs, dim=1).item()
            func_correct = (func_pred == sample['correct_function_idx'])
            # Only count function accuracy if file prediction was correct
            if file_correct and func_correct:
                function_correct_count += 1
                
            line_probs, _ = self.line_agent(bug_emb, line_emb)
            line_pred = torch.argmax(line_probs, dim=1).item()
            # Check if predicted line is within tolerance of any correct line index
            line_correct = any(abs(line_pred - idx) <= self.line_tolerance for idx in sample['correct_line_idx'])
            # Only count line accuracy if file and function predictions were correct
            if file_correct and func_correct and line_correct:
                line_correct_count += 1
                
            if file_correct and func_correct and line_correct:
                overall_correct_count += 1

        metrics = {
            'file_accuracy': file_correct_count / total_samples,
            'function_accuracy': function_correct_count / total_samples,
            'line_accuracy': line_correct_count / total_samples,
            'overall_accuracy': overall_correct_count / total_samples
        }
        return metrics

    def train(self, epochs):
        self.pretrain_all()
        self.file_agent.train()
        self.function_agent.train()
        self.line_agent.train()
        total_rewards = 0
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_reward = 0
            total_correct_samples = 0
            for sample in self.dataset:
                try:
                    loss, reward = self.train_episode(sample)
                    epoch_loss += loss.item()
                    epoch_reward += reward
                    self.file_optimizer.zero_grad()
                    self.func_optimizer.zero_grad()
                    self.line_optimizer.zero_grad()
                    loss.backward()
                    self.file_optimizer.step()
                    self.func_optimizer.step()
                    self.line_optimizer.step()
                    total_correct_samples += 1
                except Exception as e:
                    # Optionally log the error
                    continue
            try:
                print("Total processed samples:", total_correct_samples, "of", len(self.dataset))
                print("Evaluation Metrics:", self.compute_metrics())
                avg_loss = epoch_loss / len(self.dataset)
                avg_reward = epoch_reward / len(self.dataset)
                if isinstance(avg_reward, torch.Tensor):
                    avg_reward = avg_reward.item()
                if isinstance(avg_loss, torch.Tensor):
                    avg_loss = avg_loss.item()
                print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}, Average Reward = {avg_reward:.4f}")
                total_rewards += epoch_reward
            except Exception as e:
                print("Error during evaluation:", e)
                continue
        print(f"Total Reward after {epochs} epochs: {total_rewards/epochs:.4f}")
        metrics = self.compute_metrics()
        print("Final Evaluation Metrics:", metrics)


# --- Hyperparameters and Instantiation ---

bug_emb_dim = 768
file_emb_dim = 768
func_emb_dim = 768
line_emb_dim = 768
hidden_dim = 128

from atlas_agent import FileLevelAgent, FunctionLevelAgent, LineLevelAgent
from dataembedder import CodeBERTEmbedder

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = CodeBERTEmbedder(device=device)
file_agent = FileLevelAgent(bug_emb_dim, file_emb_dim, hidden_dim, use_projection=True, projection_dim=128)
function_agent = FunctionLevelAgent(bug_emb_dim, func_emb_dim, hidden_dim, use_projection=True, projection_dim=128)
line_agent = LineLevelAgent(bug_emb_dim, line_emb_dim, hidden_dim, use_projection=True, projection_dim=128)
file_optimizer = optim.Adam(file_agent.parameters(), lr=1e-3)
func_optimizer = optim.Adam(function_agent.parameters(), lr=1e-3)
line_optimizer = optim.Adam(line_agent.parameters(), lr=1e-3)

from data_loader import JsonDataset
train_data = JsonDataset('/Users/rishavsinha/Documents/atlas/dataset/train-001.json')
test_data = JsonDataset('/Users/rishavsinha/Documents/atlas/dataset/val.json')

trainer = HRLTrainer(
    train_data=train_data,
    test_data=test_data, 
    file_agent=file_agent,
    function_agent=function_agent,
    line_agent=line_agent,
    file_optimizer=file_optimizer,
    func_optimizer=func_optimizer,
    line_optimizer=line_optimizer,
    device="cpu",
    entropy_coef=0.01,
    reward_mode="intermediate",  # Options: "sparse", "intermediate", "ranking", "mixed"
    alpha=0.5,
    tau=0.1,
    file_pretrain_epochs=0,
    func_pretrain_epochs=0,
    line_pretrain_epochs=0,
    line_tolerance=2  # Tolerance threshold for line predictions
)

trainer.train(epochs=100)
