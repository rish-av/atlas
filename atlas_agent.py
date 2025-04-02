import torch
import torch.nn as nn
import torch.nn.functional as F

# File-level agent using LSTM to rank candidate files
class FileLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, file_emb_dim, hidden_dim):
        super(FileLevelAgent, self).__init__()
        self.input_dim = bug_emb_dim + file_emb_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, bug_emb, file_candidates_emb):
        batch_size = bug_emb.size(0)
        num_candidates = file_candidates_emb.size(1)
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        lstm_input = torch.cat([bug_expanded, file_candidates_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        scores = self.fc(lstm_out).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        return probs, scores

class FunctionLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, func_emb_dim, hidden_dim):
        super(FunctionLevelAgent, self).__init__()
        self.input_dim = bug_emb_dim + func_emb_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, bug_emb, func_candidates_emb):
        batch_size = bug_emb.size(0)
        num_candidates = func_candidates_emb.size(1)
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        lstm_input = torch.cat([bug_expanded, func_candidates_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        scores = self.fc(lstm_out).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        return probs, scores

class LineLevelAgent(nn.Module):
    def __init__(self, bug_emb_dim, line_emb_dim, hidden_dim):
        super(LineLevelAgent, self).__init__()
        self.input_dim = bug_emb_dim + line_emb_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, bug_emb, line_candidates_emb):
        batch_size = bug_emb.size(0)
        num_candidates = line_candidates_emb.size(1)
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        lstm_input = torch.cat([bug_expanded, line_candidates_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        scores = self.fc(lstm_out).squeeze(-1)
        probs = torch.sigmoid(scores)
        return probs, scores