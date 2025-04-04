import torch
import torch.nn as nn
import torch.optim as optim
# --- HRLTrainer with Pretraining Options and Choice of Reward Modes ---

class HRLTrainer:
    def __init__(self, train_data, test_data, embedder, file_agent, function_agent, line_agent, 
                 file_optimizer, func_optimizer, line_optimizer, device="cpu", 
                 entropy_coef=0.01, reward_mode="sparse", alpha=0.5, tau=0.1,
                 file_pretrain_epochs=0, func_pretrain_epochs=0, line_pretrain_epochs=0):
        """
        reward_mode: "sparse", "intermediate", "ranking", or "mixed"
        alpha: weight for intermediate reward in mixed mode (0<=alpha<=1)
        tau: temperature parameter for ranking reward approximations
        Pretrain epochs: number of epochs to pretrain each agent individually.
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
        self.file_pretrain_epochs = file_pretrain_epochs
        self.func_pretrain_epochs = func_pretrain_epochs
        self.line_pretrain_epochs = line_pretrain_epochs

    def pretrain_file_agent(self):
        self.file_agent.train()
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(self.file_pretrain_epochs):
            epoch_loss = 0
            for sample in self.dataset:
                bug_emb = self.embedder.get_bug_embedding(sample['bug_report']).to(self.device).unsqueeze(0)
                file_candidates = sample['candidate_files']
                file_emb = self.embedder.get_file_embeddings(file_candidates).to(self.device).unsqueeze(0)
                file_probs, _ = self.file_agent(bug_emb, file_emb)
                target = torch.tensor([sample['correct_file_idx']], dtype=torch.long, device=self.device)
                loss = loss_fn(file_probs, target)
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
                bug_emb = self.embedder.get_bug_embedding(sample['bug_report']).to(self.device).unsqueeze(0)
                func_candidates = sample['candidate_functions']
                func_emb = self.embedder.get_function_embeddings(func_candidates).to(self.device).unsqueeze(0)
                func_probs, _ = self.function_agent(bug_emb, func_emb)
                target = torch.tensor([sample['correct_function_idx']], dtype=torch.long, device=self.device)
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
                bug_emb = self.embedder.get_bug_embedding(sample['bug_report']).to(self.device).unsqueeze(0)
                line_candidates = sample['candidate_lines']
                line_emb = self.embedder.get_line_embeddings(line_candidates).to(self.device).unsqueeze(0)
                line_probs, _ = self.line_agent(bug_emb, line_emb)
                # Assume that during pretraining, sample has a single integer target for the correct line.
                target = torch.tensor([sample['correct_line_idx']], dtype=torch.long, device=self.device)
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

    def _compute_reward_line(self, line_probs, true_labels):
        # When using CE, we assume a single correct index. Here, for reward computation, we compare the probability
        # assigned to the correct line.
        # In practice, you could also use a ranking surrogate similar to file/function levels.
        return line_probs[0, true_labels]

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
        
        line_probs, line_scores = self.line_agent(bug_emb, line_emb)
        m_line = torch.distributions.Categorical(line_probs)
        line_action = m_line.sample()
        log_prob_line = m_line.log_prob(line_action)
        
        if self.reward_mode == "sparse":
            file_correct = (file_action.item() == sample['correct_file_idx'])
            func_correct = (func_action.item() == sample['correct_function_idx'])
            line_correct = (line_action.item() == sample['correct_line_idx'])
            reward = 1.0 if file_correct and func_correct and line_correct else 0.0
        elif self.reward_mode in ["intermediate", "ranking", "mixed"]:
            r_file_int = self._compute_intermediate_reward_file(file_probs, file_scores, sample['correct_file_idx'])
            r_func_int = self._compute_intermediate_reward_func(func_probs, func_scores, sample['correct_function_idx'])
            # For line-level, assume a single correct index is provided in sample['correct_line_idx']
            target_line_idx = sample['correct_line_idx']
            r_line_int = line_probs[0, target_line_idx]
            r_file_rank = self._compute_ranking_reward_file(file_scores, sample['correct_file_idx'])
            r_func_rank = self._compute_ranking_reward_func(func_scores, sample['correct_function_idx'])
            # For line, use the intermediate reward as the ranking reward
            r_line_rank = r_line_int
            if self.reward_mode == "intermediate":
                reward = (r_file_int + r_func_int + r_line_int) / 3.0
            elif self.reward_mode == "ranking":
                reward = (r_file_rank + r_func_rank + r_line_rank) / 3.0
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
            line_pred = torch.argmax(line_probs, dim=1).item()
            if line_pred == sample['correct_line_idx']:
                line_correct_count += 1
                
            if (file_pred == sample['correct_file_idx'] and 
                func_pred == sample['correct_function_idx'] and 
                line_pred == sample['correct_line_idx']):
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
            for sample in self.dataset:
                print(f"Training on sample: {sample['bug_report']}")
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
            print("Evaluation Metrics:", self.compute_metrics())
        print(f"Total Reward after {epochs} epochs: {total_rewards/epochs:.4f}")
        metrics = self.compute_metrics()
        print("Evaluation Metrics:", metrics)



from sample_data import generate_semantic_dataset
data = generate_semantic_dataset(200)
train_data = data[:180]
test_data = data[180:]

bug_emb_dim = 768
file_emb_dim = 768
func_emb_dim = 768
line_emb_dim = 768
hidden_dim = 128

from atlas_agent import FileLevelAgent, FunctionLevelAgent, LineLevelAgent
from dataembedder import CodeBERTEmbedder

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = CodeBERTEmbedder(device=device)
file_agent = FileLevelAgent(bug_emb_dim, file_emb_dim, hidden_dim)
function_agent = FunctionLevelAgent(bug_emb_dim, func_emb_dim, hidden_dim)
line_agent = LineLevelAgent(bug_emb_dim, line_emb_dim, hidden_dim)
file_optimizer = optim.Adam(file_agent.parameters(), lr=1e-3)
func_optimizer = optim.Adam(function_agent.parameters(), lr=1e-3)
line_optimizer = optim.Adam(line_agent.parameters(), lr=1e-3)

from sample_data import generate_semantic_dataset
train_data = generate_semantic_dataset(500)
test_data = generate_semantic_dataset(25)

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
    reward_mode="sparse",  # Options: "sparse", "intermediate", "ranking", "mixed"
    alpha=0.5,
    tau=0.1,
    file_pretrain_epochs=5,
    func_pretrain_epochs=5,
    line_pretrain_epochs=5
)

trainer.train(epochs=100)
