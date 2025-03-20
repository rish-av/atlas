import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModel

class CodeBERTEmbedder:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, code: str) -> np.ndarray:
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

class BugLocalizationEnv:
    def __init__(self, embedder: CodeBERTEmbedder):
        self.embedder = embedder
        self.project = self._generate_dummy_project()
        self.bug_location = self._select_bug_location()
        self.current_file = None
        self.current_function = None

    def _generate_dummy_project(self):
        project = []
        for f in range(5):
            file_code = f"// File {f} code\n" + "\n".join([f"line {i}" for i in range(100)])
            functions = []
            for func in range(3):
                func_code = f"def func_{func}():\n" + "\n".join([f"    statement {i}" for i in range(10)])
                lines = [f"statement {i}" for i in range(10)]
                functions.append({"code": func_code, "lines": lines})
            project.append({"code": file_code, "functions": functions, "struct_features": np.random.rand(1, 64)})
        return project

    def _select_bug_location(self):
        file_idx = random.randint(0, len(self.project) - 1)
        func_idx = random.randint(0, len(self.project[file_idx]["functions"]) - 1)
        line_idx = random.randint(0, len(self.project[file_idx]["functions"][func_idx]["lines"]) - 1)
        return (file_idx, func_idx, line_idx)

    def get_stack_trace_vector(self) -> np.ndarray:
        return np.random.rand(1, 128)

    def reset_file_level(self) -> np.ndarray:
        self.current_file = None
        stack_trace = self.get_stack_trace_vector()
        file_states = []
        for file in self.project:
            e_code = self.embedder.get_embedding(file["code"])
            e_struct = file["struct_features"]
            ef = np.concatenate([e_code, e_struct], axis=1)
            state = np.concatenate([ef, stack_trace], axis=1)
            file_states.append(state.squeeze(0))
        return np.array(file_states)

    def step_file_level(self, action: int):
        self.current_file = self.project[action]
        bug_file = self.bug_location[0]
        reward = 1.0 if action == bug_file else -0.1
        done = (action == bug_file)
        return reward, done, action

    def get_function_states(self) -> np.ndarray:
        assert self.current_file is not None
        stack_trace = self.get_stack_trace_vector()
        function_states = []
        for func in self.current_file["functions"]:
            e_func = self.embedder.get_embedding(func["code"])
            state = np.concatenate([e_func, stack_trace], axis=1)
            function_states.append(state.squeeze(0))
        return np.array(function_states)

    def step_function_level(self, action: int):
        self.current_function = self.current_file["functions"][action]
        bug_func = self.bug_location[1]
        reward = 1.0 if action == bug_func else -0.1
        done = (action == bug_func)
        return reward, done, action

    def get_line_states(self) -> np.ndarray:
        assert self.current_function is not None
        stack_trace = self.get_stack_trace_vector()
        line_states = []
        for line in self.current_function["lines"]:
            e_line = self.embedder.get_embedding(line)
            state = np.concatenate([e_line, stack_trace], axis=1)
            line_states.append(state.squeeze(0))
        return np.array(line_states)

    def step_line_level(self, action: int):
        bug_line = self.bug_location[2]
        reward = 1.0 if action == bug_line else -0.1
        done = (action == bug_line)
        return reward, done, action
