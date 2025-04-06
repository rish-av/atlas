import torch
from transformers import RobertaTokenizer, RobertaModel
import re
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


def extract_functions_regex(file_content):
    try:
        method_pattern = r'((?:public|private|protected|static|final|abstract|synchronized)\s+[\w\<\>\[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{)'
        functions = []
        function_info = []
        for match in re.finditer(method_pattern, file_content):
            method_start = match.start()
            method_name = match.group(2)
            start_line = file_content[:method_start].count('\n') + 1
            open_braces = 0
            close_pos = match.end()
            for i in range(match.end(), len(file_content)):
                if file_content[i] == '{':
                    open_braces += 1
                elif file_content[i] == '}':
                    if open_braces == 0:
                        close_pos = i + 1
                        break
                    open_braces -= 1
            end_line = file_content[:close_pos].count('\n') + 1
            function_content = file_content[match.start():close_pos]
            functions.append(function_content)
            function_info.append({'name': method_name, 'start_line': start_line, 'end_line': end_line})
        return functions, function_info
    except Exception as e:
        print(f"Error extracting functions: {e}")
        return [], []

def extract_global_code(file_content, function_info):
    try:
        lines = file_content.split('\n')
        function_line_ranges = [(info['start_line'], info['end_line']) for info in function_info]
        global_lines = []
        for i, line in enumerate(lines, 1):
            if not any(start <= i <= end for start, end in function_line_ranges):
                global_lines.append(line)
        return '\n'.join(global_lines)
    except Exception as e:
        print(f"Error extracting global code: {e}")
        return ""
