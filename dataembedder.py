import torch
from transformers import RobertaTokenizer, RobertaModel

class CodeBERTEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding

    def get_bug_embedding(self, bug_report: str) -> torch.Tensor:
        return self.embed_text(bug_report)

    def get_file_embeddings(self, candidate_files: list) -> torch.Tensor:
        embeddings = [self.embed_text(file_content) for file_content in candidate_files]
        return torch.stack(embeddings, dim=0)

    def get_function_embeddings(self, candidate_functions: list) -> torch.Tensor:
        embeddings = [self.embed_text(func_content) for func_content in candidate_functions]
        return torch.stack(embeddings, dim=0)

    def get_line_embeddings(self, candidate_lines: list) -> torch.Tensor:
        embeddings = [self.embed_text(line) for line in candidate_lines]
        return torch.stack(embeddings, dim=0)
