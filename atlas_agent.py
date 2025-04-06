import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple MLP used before LSTM.
class PreLSTMMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(PreLSTMMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class AttentionPool(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPool, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(attn_weights * x, dim=1)         # (batch, input_dim)
        return pooled

class FileLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, file_emb_dim, lstm_hidden_dim, 
                 mlp_hidden_dim=128, mlp_out_dim=128, use_projection=False, 
                 projection_dim=None, use_hierarchical_pool=True):
        super(FileLevelAgent, self).__init__()
        self.input_dim = bug_emb_dim + file_emb_dim
        self.mlp = PreLSTMMLP(self.input_dim, mlp_hidden_dim, mlp_out_dim)
        self.use_projection = use_projection
        if self.use_projection:
            if projection_dim is None:
                projection_dim = mlp_out_dim
            self.projection = nn.Linear(mlp_out_dim, projection_dim)
            lstm_input_dim = projection_dim
        else:
            lstm_input_dim = mlp_out_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, 1)
        self.use_hierarchical_pool = use_hierarchical_pool
        if self.use_hierarchical_pool:
            self.attention_pool = AttentionPool(file_emb_dim)
    
    def forward(self, bug_emb, file_candidates_emb):
        # bug_emb: (batch, bug_emb_dim) or (batch, 1, bug_emb_dim)
        # file_candidates_emb: either (batch, num_candidates, file_emb_dim)
        #    or (batch, num_candidates, seq_len, file_emb_dim) if hierarchical pooling is used.
        if bug_emb.dim() > 2:
            bug_emb = bug_emb.mean(dim=1)
        if self.use_hierarchical_pool and file_candidates_emb.dim() == 4:
            batch, num_candidates, seq_len, emb_dim = file_candidates_emb.size()
            file_candidates_emb = file_candidates_emb.view(batch * num_candidates, seq_len, emb_dim)
            file_candidates_emb = self.attention_pool(file_candidates_emb)
            file_candidates_emb = file_candidates_emb.view(batch, num_candidates, emb_dim)
        elif file_candidates_emb.dim() == 4:
            file_candidates_emb = file_candidates_emb.mean(dim=2)
        
        num_candidates = file_candidates_emb.size(1)
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        x = torch.cat([bug_expanded, file_candidates_emb], dim=-1)
        x = self.mlp(x)
        if self.use_projection:
            x = self.projection(x)
        lstm_out, _ = self.lstm(x)
        scores = self.fc(lstm_out).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        return probs, scores

class FunctionLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, func_emb_dim, 
                 mlp_hidden_dim=128, mlp_out_dim=128, 
                 use_projection=False, projection_dim=None, 
                 use_hierarchical_pool=True):
        super(FunctionLevelAgent, self).__init__()
        self.input_dim = bug_emb_dim + func_emb_dim
        self.mlp = PreLSTMMLP(self.input_dim, mlp_hidden_dim, mlp_out_dim)
        self.use_projection = use_projection
        if self.use_projection:
            if projection_dim is None:
                projection_dim = mlp_out_dim
            self.projection = nn.Linear(mlp_out_dim, projection_dim)
        # Instead of using an LSTM, we now directly map the MLP output to a score.
        self.fc = nn.Linear(mlp_out_dim, 1)
        self.use_hierarchical_pool = use_hierarchical_pool
        if self.use_hierarchical_pool:
            self.attention_pool = AttentionPool(func_emb_dim)
    
    def forward(self, bug_emb, func_candidates_emb):
        # If bug_emb is provided with extra dimensions, average over them.
        if bug_emb.dim() > 2:
            bug_emb = bug_emb.mean(dim=1)
        # If candidate embeddings are provided as sequences (4D tensor) and hierarchical pooling is enabled,
        # pool them to a single embedding per candidate.
        if self.use_hierarchical_pool and func_candidates_emb.dim() == 4:
            batch, num_candidates, seq_len, emb_dim = func_candidates_emb.size()
            func_candidates_emb = func_candidates_emb.view(batch * num_candidates, seq_len, emb_dim)
            func_candidates_emb = self.attention_pool(func_candidates_emb)
            func_candidates_emb = func_candidates_emb.view(batch, num_candidates, emb_dim)
        elif func_candidates_emb.dim() == 4:
            func_candidates_emb = func_candidates_emb.mean(dim=2)
        
        num_candidates = func_candidates_emb.size(1)
        # Expand the bug embedding to match the number of candidates.
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        x = torch.cat([bug_expanded, func_candidates_emb], dim=-1)
        x = self.mlp(x)
        if self.use_projection:
            x = self.projection(x)
        # Compute a score for each candidate without any sequential modeling.
        scores = self.fc(x).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        return probs, scores


class LineLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, line_emb_dim, lstm_hidden_dim, 
                 mlp_hidden_dim=128, mlp_out_dim=128, use_projection=False, 
                 projection_dim=None, use_hierarchical_pool=True):
        super(LineLevelAgent, self).__init__()
        self.input_dim = bug_emb_dim + line_emb_dim
        self.mlp = PreLSTMMLP(self.input_dim, mlp_hidden_dim, mlp_out_dim)
        self.use_projection = use_projection
        if self.use_projection:
            if projection_dim is None:
                projection_dim = mlp_out_dim
            self.projection = nn.Linear(mlp_out_dim, projection_dim)
            lstm_input_dim = projection_dim
        else:
            lstm_input_dim = mlp_out_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, 1)
        self.use_hierarchical_pool = use_hierarchical_pool
        if self.use_hierarchical_pool:
            self.attention_pool = AttentionPool(line_emb_dim)
    
    def forward(self, bug_emb, line_candidates_emb):
        if bug_emb.dim() > 2:
            bug_emb = bug_emb.mean(dim=1)
        if self.use_hierarchical_pool and line_candidates_emb.dim() == 4:
            batch, num_candidates, seq_len, emb_dim = line_candidates_emb.size()
            line_candidates_emb = line_candidates_emb.view(batch * num_candidates, seq_len, emb_dim)
            line_candidates_emb = self.attention_pool(line_candidates_emb)
            line_candidates_emb = line_candidates_emb.view(batch, num_candidates, emb_dim)
        elif line_candidates_emb.dim() == 4:
            line_candidates_emb = line_candidates_emb.mean(dim=2)
        
        num_candidates = line_candidates_emb.size(1)
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        x = torch.cat([bug_expanded, line_candidates_emb], dim=-1)
        x = self.mlp(x)
        if self.use_projection:
            x = self.projection(x)
        lstm_out, _ = self.lstm(x)
        scores = self.fc(lstm_out).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        return probs, scores

class Critic(nn.Module):
    def __init__(self, bug_emb_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(bug_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, bug_emb):
        x = F.relu(self.fc1(bug_emb))
        value = self.fc2(x)
        return value
