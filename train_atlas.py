import torch
import torch.nn as nn
import torch.optim as optim


class HRLTrainer:
    def __init__(self, dataset, embedder, file_agent, function_agent, line_agent, 
                 file_optimizer, func_optimizer, line_optimizer, device="cpu", entropy_coef=0.01):
        self.dataset = dataset
        self.embedder = embedder
        self.file_agent = file_agent.to(device)
        self.function_agent = function_agent.to(device)
        self.line_agent = line_agent.to(device)
        self.file_optimizer = file_optimizer
        self.func_optimizer = func_optimizer
        self.line_optimizer = line_optimizer
        self.device = device
        self.entropy_coef = entropy_coef
        
    def train_episode(self, sample):
        bug_emb = self.embedder.get_bug_embedding(sample['bug_report']).to(self.device).unsqueeze(0)
        file_candidates = sample['candidate_files']
        file_emb = self.embedder.get_file_embeddings(file_candidates).to(self.device).unsqueeze(0)
        func_candidates = sample['candidate_functions']
        func_emb = self.embedder.get_function_embeddings(func_candidates).to(self.device).unsqueeze(0)
        line_candidates = sample['candidate_lines']
        line_emb = self.embedder.get_line_embeddings(line_candidates).to(self.device).unsqueeze(0)
        
        file_probs, _ = self.file_agent(bug_emb, file_emb)
        m_file = torch.distributions.Categorical(file_probs)
        file_action = m_file.sample()
        log_prob_file = m_file.log_prob(file_action)
        
        func_probs, _ = self.function_agent(bug_emb, func_emb)
        m_func = torch.distributions.Categorical(func_probs)
        func_action = m_func.sample()
        log_prob_func = m_func.log_prob(func_action)
        
        line_probs, _ = self.line_agent(bug_emb, line_emb)
        m_line = torch.distributions.Bernoulli(probs=line_probs)
        line_action = m_line.sample()
        log_prob_line = m_line.log_prob(line_action).sum()
        
        file_correct = (file_action.item() == sample['correct_file_idx'])
        func_correct = (func_action.item() == sample['correct_function_idx'])
        true_line_labels = torch.tensor(sample['correct_line_labels'], dtype=torch.float32).to(self.device)
        lines_correct = torch.allclose(line_action.squeeze(), true_line_labels, atol=0.5)
        
        reward = 1.0 if file_correct and func_correct and lines_correct else 0.0
        
        total_log_prob = log_prob_file + log_prob_func + log_prob_line
        entropy = m_file.entropy() + m_func.entropy() + m_line.entropy().sum()
        loss = - total_log_prob * reward - self.entropy_coef * entropy
        return loss, reward

    def train(self, epochs):
        self.file_agent.train()
        self.function_agent.train()
        self.line_agent.train()
        total_rewards = 0
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_reward = 0
            for sample in self.dataset:
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
            avg_loss = epoch_loss / len(self.dataset)
            avg_reward = epoch_reward / len(self.dataset)
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, Average Reward = {avg_reward:.4f}")
            total_rewards += epoch_reward
        print(f"Total Reward after {epochs} epochs: {total_rewards:.4f}")



from sample_data import sample_data
sample_data_obj = sample_data()
train_data = sample_data_obj.train_dataset
test_data = sample_data_obj.test_dataset

bug_emb_dim = 768
file_emb_dim = 768
func_emb_dim = 768
line_emb_dim = 768
hidden_dim = 128

from atlas_agent import FileLevelAgent, FunctionLevelAgent, LineLevelAgent
from dataembedder import CodeBERTEmbedder

embedder = CodeBERTEmbedder()
file_agent = FileLevelAgent(bug_emb_dim, file_emb_dim, hidden_dim)
function_agent = FunctionLevelAgent(bug_emb_dim, func_emb_dim, hidden_dim)
line_agent = LineLevelAgent(bug_emb_dim, line_emb_dim, hidden_dim)
file_optimizer = optim.Adam(file_agent.parameters(), lr=1e-3)
func_optimizer = optim.Adam(function_agent.parameters(), lr=1e-3)
line_optimizer = optim.Adam(line_agent.parameters(), lr=1e-3)

trainer = HRLTrainer(
    dataset=train_data,
    embedder=embedder,
    file_agent=file_agent,
    function_agent=function_agent,
    line_agent=line_agent,
    file_optimizer=file_optimizer,
    func_optimizer=func_optimizer,
    line_optimizer=line_optimizer,
    device="cpu",
    entropy_coef=0.01
)

trainer.train(epochs=10)
