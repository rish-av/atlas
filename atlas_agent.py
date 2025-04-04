import torch
import torch.nn as nn
import torch.nn.functional as F

# File-level agent using LSTM to rank candidate files
class PreLSTMMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(PreLSTMMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# File-level agent with an additional MLP before LSTM
class FileLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, file_emb_dim, lstm_hidden_dim, mlp_hidden_dim=128, mlp_out_dim=128, use_projection=False, projection_dim=None):
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
    
    def forward(self, bug_emb, file_candidates_emb):
        batch_size = bug_emb.size(0)
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

# Function-level agent with extra linear layers before LSTM
class FunctionLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, func_emb_dim, lstm_hidden_dim, mlp_hidden_dim=128, mlp_out_dim=128, use_projection=False, projection_dim=None):
        super(FunctionLevelAgent, self).__init__()
        self.input_dim = bug_emb_dim + func_emb_dim
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
    
    def forward(self, bug_emb, func_candidates_emb):
        batch_size = bug_emb.size(0)
        num_candidates = func_candidates_emb.size(1)
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        x = torch.cat([bug_expanded, func_candidates_emb], dim=-1)
        x = self.mlp(x)
        if self.use_projection:
            x = self.projection(x)
        lstm_out, _ = self.lstm(x)
        scores = self.fc(lstm_out).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        return probs, scores

# Line-level agent with extra linear layers before LSTM; here we use softmax and CE loss.
class LineLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, line_emb_dim, lstm_hidden_dim, mlp_hidden_dim=128, mlp_out_dim=128, use_projection=False, projection_dim=None):
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
    
    def forward(self, bug_emb, line_candidates_emb):
        batch_size = bug_emb.size(0)
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