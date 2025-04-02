import torch
import torch.nn as nn
import torch.optim as optim


class HRLTrainer:
    def __init__(self,  train_data, test_data, embedder, file_agent, function_agent, line_agent, 
                 file_optimizer, func_optimizer, line_optimizer, device="cpu", 
                 entropy_coef=0.01, reward_mode="sparse", alpha=0.5, tau=0.1):
        """
        reward_mode: "sparse", "intermediate", "ranking", or "mixed"
        alpha: weight for intermediate reward in mixed mode (0<=alpha<=1)
        tau: temperature parameter for ranking reward approximations
        """
        self.dataset = train_data
        self.test_dataset = test_data
        self.embedder = embedder
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
        
    def _compute_intermediate_reward_file(self, file_probs, file_scores, correct_idx):
        r = file_probs[0, correct_idx]
        return r

    def _compute_ranking_reward_file(self, file_scores, correct_idx):
        score_correct = file_scores[0, correct_idx]
        diff = file_scores[0] - score_correct
        mask = torch.ones_like(diff, dtype=torch.bool)
        mask[correct_idx] = False
        rank_approx = 1 + torch.sum(torch.sigmoid(diff[mask] / self.tau))
        r = 1 / torch.log2(rank_approx + 1)
        return r

    def _compute_intermediate_reward_func(self, func_probs, func_scores, correct_idx):
        r = func_probs[0, correct_idx]
        return r

    def _compute_ranking_reward_func(self, func_scores, correct_idx):
        score_correct = func_scores[0, correct_idx]
        diff = func_scores[0] - score_correct
        mask = torch.ones_like(diff, dtype=torch.bool)
        mask[correct_idx] = False
        rank_approx = 1 + torch.sum(torch.sigmoid(diff[mask] / self.tau))
        r = 1 / torch.log2(rank_approx + 1)
        return r

    def _compute_reward_line(self, line_probs, true_labels):
        # Use 1 minus mean absolute error as reward for line level.
        error = torch.mean(torch.abs(line_probs.squeeze() - true_labels))
        r = 1 - error
        return r

    def train_episode(self, sample):
        bug_emb = self.embedder.get_bug_embedding(sample['bug_report']).to(self.device).unsqueeze(0)
        file_candidates = sample['candidate_files']
        file_emb = self.embedder.get_file_embeddings(file_candidates).to(self.device).unsqueeze(0)
        func_candidates = sample['candidate_functions']
        func_emb = self.embedder.get_function_embeddings(func_candidates).to(self.device).unsqueeze(0)
        line_candidates = sample['candidate_lines']
        line_emb = self.embedder.get_line_embeddings(line_candidates).to(self.device).unsqueeze(0)
        
        file_probs, file_scores = self.file_agent(bug_emb, file_emb)
        m_file = torch.distributions.Categorical(file_probs)
        file_action = m_file.sample()
        log_prob_file = m_file.log_prob(file_action)
        
        func_probs, func_scores = self.function_agent(bug_emb, func_emb)
        m_func = torch.distributions.Categorical(func_probs)
        func_action = m_func.sample()
        log_prob_func = m_func.log_prob(func_action)
        
        line_probs, _ = self.line_agent(bug_emb, line_emb)
        m_line = torch.distributions.Bernoulli(probs=line_probs)
        line_action = m_line.sample()
        log_prob_line = m_line.log_prob(line_action).sum()
        
        # Compute reward depending on mode
        if self.reward_mode == "sparse":
            file_correct = (file_action.item() == sample['correct_file_idx'])
            func_correct = (func_action.item() == sample['correct_function_idx'])
            true_line_labels = torch.tensor(sample['correct_line_labels'], dtype=torch.float32).to(self.device)
            lines_correct = torch.allclose(line_action.squeeze(), true_line_labels, atol=0.5)
            reward = 1.0 if file_correct and func_correct and lines_correct else 0.0
        elif self.reward_mode in ["intermediate", "ranking", "mixed"]:
            # Intermediate rewards
            r_file_int = self._compute_intermediate_reward_file(file_probs, file_scores, sample['correct_file_idx'])
            r_func_int = self._compute_intermediate_reward_func(func_probs, func_scores, sample['correct_function_idx'])
            true_line_labels = torch.tensor(sample['correct_line_labels'], dtype=torch.float32).to(self.device)
            r_line = self._compute_reward_line(line_probs, true_line_labels)
            # Ranking rewards (for file and function levels)
            r_file_rank = self._compute_ranking_reward_file(file_scores, sample['correct_file_idx'])
            r_func_rank = self._compute_ranking_reward_func(func_scores, sample['correct_function_idx'])
            # For line level, we use the same as intermediate reward
            r_line_rank = r_line
            if self.reward_mode == "intermediate":
                reward = (r_file_int + r_func_int + r_line) / 3.0
            elif self.reward_mode == "ranking":
                reward = (r_file_rank + r_func_rank + r_line_rank) / 3.0
            elif self.reward_mode == "mixed":
                reward_int = (r_file_int + r_func_int + r_line) / 3.0
                reward_rank = (r_file_rank + r_func_rank + r_line_rank) / 3.0
                reward = self.alpha * reward_int + (1 - self.alpha) * reward_rank
        else:
            reward = 0.0

        total_log_prob = log_prob_file + log_prob_func + log_prob_line
        entropy = m_file.entropy() + m_func.entropy() + m_line.entropy().sum()
        loss = - total_log_prob * reward - self.entropy_coef * entropy
        return loss, reward

    def compute_metrics(self):
        total_samples = len(self.dataset)
        file_correct_count = 0
        function_correct_count = 0
        line_correct_count = 0
        overall_correct_count = 0
        for sample in self.dataset:
            bug_emb = self.embedder.get_bug_embedding(sample['bug_report']).to(self.device).unsqueeze(0)
            file_candidates = sample['candidate_files']
            file_emb = self.embedder.get_file_embeddings(file_candidates).to(self.device).unsqueeze(0)
            func_candidates = sample['candidate_functions']
            func_emb = self.embedder.get_function_embeddings(func_candidates).to(self.device).unsqueeze(0)
            line_candidates = sample['candidate_lines']
            line_emb = self.embedder.get_line_embeddings(line_candidates).to(self.device).unsqueeze(0)
            
            file_probs, _ = self.file_agent(bug_emb, file_emb)
            file_pred = torch.argmax(file_probs, dim=1).item()
            if file_pred == sample['correct_file_idx']:
                file_correct_count += 1
                
            func_probs, _ = self.function_agent(bug_emb, func_emb)
            func_pred = torch.argmax(func_probs, dim=1).item()
            if func_pred == sample['correct_function_idx']:
                function_correct_count += 1
                
            line_probs, _ = self.line_agent(bug_emb, line_emb)
            line_pred = (line_probs > 0.5).float().squeeze()
            true_line_labels = torch.tensor(sample['correct_line_labels'], dtype=torch.float32).to(self.device)
            if torch.allclose(line_pred, true_line_labels, atol=0.5):
                line_correct_count += 1
                
            if (file_pred == sample['correct_file_idx'] and 
                func_pred == sample['correct_function_idx'] and 
                torch.allclose(line_pred, true_line_labels, atol=0.5)):
                overall_correct_count += 1

        metrics = {
            'file_accuracy': file_correct_count / total_samples,
            'function_accuracy': function_correct_count / total_samples,
            'line_accuracy': line_correct_count / total_samples,
            'overall_accuracy': overall_correct_count / total_samples
        }
        return metrics

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
            print(self.compute_metrics())
        print(f"Total Reward after {epochs} epochs: {total_rewards:.4f}")
        metrics = self.compute_metrics()
        print("Evaluation Metrics:", metrics)



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
    train_data=train_data,
    test_data=test_data,
    embedder=embedder,
    file_agent=file_agent,
    function_agent=function_agent,
    line_agent=line_agent,
    file_optimizer=file_optimizer,
    func_optimizer=func_optimizer,
    line_optimizer=line_optimizer,
    device="cpu",
    entropy_coef=0.01,
    reward_mode="intermediate",
)

trainer.train(epochs=10)
