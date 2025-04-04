import os
import json
import re
import numpy as np
import faiss
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
import git
import traceback



class CodeEmbedder:
    
    def __init__(self, model_name="microsoft/codebert-base", device=None):
      
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = 768

    
    def embed_text(self, text, max_length=512):
        if not text:
            return np.zeros(768)
            
        # Truncate very long text to avoid tokenizer errors
        if len(text) > max_length * 10:
            text = text[:max_length * 10]
            
        inputs = self.tokenizer(text, padding=True, truncation=True, 
                              max_length=max_length, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()[0]
        
        return embedding
    
    def embed_function(self, function_content):
        return self.embed_text(function_content)
    
    def embed_line(self, line_content):
        return self.embed_text(line_content)





def extract_functions_regex(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Find class name
        class_pattern = r'(?:public|private|protected)?\s+(?:abstract|final)?\s*class\s+([A-Za-z0-9_]+)'
        class_match = re.search(class_pattern, content)
        class_name = class_match.group(1) if class_match else "UnknownClass"
        
        # Find methods using regex
        method_pattern = r'((?:public|private|protected|static|final|abstract|synchronized)\s+[\w\<\>\[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{)'
        
        functions = []
        for match in re.finditer(method_pattern, content):
            method_start = match.start()
            method_name = match.group(2)
            
            # Count lines until method start
            start_line = content[:method_start].count('\n') + 1
            
            # Find end of method by matching braces
            open_braces = 0
            close_pos = match.end()
            
            for i in range(match.end(), len(content)):
                if content[i] == '{':
                    open_braces += 1
                elif content[i] == '}':
                    if open_braces == 0:
                        close_pos = i + 1
                        break
                    open_braces -= 1
                    
            # Calculate end line
            end_line = content[:close_pos].count('\n') + 1
            
            # Extract function content
            function_content = content[match.start():close_pos]
            
            # Add the function
            functions.append({
                'name': method_name,
                'class': class_name,
                'start_line': start_line,
                'end_line': end_line,
                'content': function_content
            })
            
        return functions
    except Exception as e:
        print(f"Error extracting functions from {file_path}: {e}")
        return []
    
    

def extract_lines_from_function(function, file_path, target_line_number=None):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            
        start_line = function['start_line']
        end_line = function['end_line']
        
        function_lines = []
        for i in range(start_line - 1, min(end_line, len(all_lines))):
            line_content = all_lines[i].rstrip()
            if not line_content:
                continue
                
            is_target = (i + 1 == target_line_number)
            
            if target_line_number:
                distance = abs(i + 1 - target_line_number)
            else:
                distance = float('inf')
            
            function_lines.append({
                'line_num': i + 1,
                'content': line_content,
                'is_target': is_target,
                'distance': distance
            })
        
        # Sort lines: target first, then by distance to target
        function_lines.sort(key=lambda l: (0 if l['is_target'] else 1, l['distance']))

        return function_lines
        
    except Exception as e:
        print(f"Error extracting lines from function {function['name']} in {file_path}: {e}")
        return []

def checkout_repo(repo_path, sha):
    try:
        repo = git.Repo(repo_path)
        repo.git.checkout(sha, force=True)
        return True
    except Exception as e:
        print(f"Error checking out SHA {sha} in {repo_path}: {e}")
        return False



def load_resources(data_dir):
    """Load existing embeddings, FAISS index, and metadata."""
    resources = {}
    
    embeddings_dir = data_dir / "embeddings"
    temp_dir = data_dir / "tmp"
    # Load FAISS index
    print("Loading FAISS index...")
    faiss_index_path = embeddings_dir / "faiss_index.bin"
    resources['faiss_index'] = faiss.read_index(str(faiss_index_path))
    
    # Load FAISS metadata
    print("Loading FAISS metadata...")
    with open(embeddings_dir / "faiss_metadata.json", 'r') as f:
        resources['faiss_metadata'] = json.load(f)
    
    # Load bug mapping
    print("Loading bug mapping...")
    with open(embeddings_dir / "bug_mapping.json", 'r') as f:
        resources['bug_mapping'] = json.load(f)
    
    # Load buggy file embeddings
    print("Loading buggy file embeddings...")
    resources['buggy_file_embeddings'] = np.load(temp_dir / "buggy_file_embeddings.npy")
    
    # Load trace embeddings
    print("Loading trace embeddings...")
    resources['trace_embeddings'] = np.load(temp_dir / "trace_embeddings.npy")
    
    # Create lookup for bug ID to embedding indices
    resources['bug_id_to_embedding'] = {}
    for i, bug_data in enumerate(resources['bug_mapping']):
        bug_id = bug_data['metadata']['bug_id']
        resources['bug_id_to_embedding'][bug_id] = {
            'file_embedding_index': bug_data['file_embedding_index'],
            'trace_embedding_index': bug_data['trace_embedding_index']
        }
    
    return resources



def search_similar_files(resources, bug_id, top_k=3):
    """Search for similar files using the bug's stack trace embedding."""
    # Get the trace embedding for this bug
    if bug_id in resources['bug_id_to_embedding']:
        trace_idx = resources['bug_id_to_embedding'][bug_id]['trace_embedding_index']
        trace_embedding = resources['trace_embeddings'][trace_idx]
        
        # Prepare for FAISS search
        query_vector = np.array([trace_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = resources['faiss_index'].search(query_vector, top_k)
        
        # Extract results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(resources['faiss_metadata']):
                file_info = resources['faiss_metadata'][idx]
                results.append({
                    'index': int(idx),
                    'repo': file_info['repo'],
                    'path': file_info['path'],
                    'distance': float(distances[0][i])
                })
        
        return results
    else:
        print(f"Bug ID {bug_id} not found in embedding lookup")
        return []



def create_hierarchical_dataset(bugs, resources, embedder):
    """Create the hierarchical bug localization dataset."""
    dataset = []
    
    for bug in tqdm(bugs, desc="Processing bugs"):
        try:
            # Extract bug information
            bug_id = bug['issue_id']
            repo_name = bug['repo_name']
            repo_simple_name = repo_name.split('/')[-1]
            buggy_file_path = f"{bug['path_to_buggy_file']}/{bug['buggy_file_name']}"
            buggy_line_number = bug['buggy_line_number']
            buggy_function_name = bug.get('buggy_function_name', '')
            
            # Get repository path
            repo_path = os.path.join("../data/repos", repo_simple_name)
            
            # Search for similar files
            similar_files = search_similar_files(resources, bug_id)
            # Check if buggy file is in results
            buggy_file_in_results = False
            correct_file_idx = -1
            
            for i, file_info in enumerate(similar_files):
                if file_info['repo'] == repo_name and file_info['path'] == buggy_file_path:
                    buggy_file_in_results = True
                    correct_file_idx = i
                    break
            
            # If buggy file is not in results, add it
            if not buggy_file_in_results:
                # Get the file embedding for this bug
                if bug_id in resources['bug_id_to_embedding']:
                    file_idx = resources['bug_id_to_embedding'][bug_id]['file_embedding_index']
                    similar_files.append({
                        'index': -1,  # Special index for added file
                        'repo': repo_name,
                        'path': buggy_file_path,
                        'distance': 1.0  # Default distance
                    })
                    correct_file_idx = len(similar_files) - 1
            
            # Initialize data structures
            candidate_files = []
            file_embeddings = []
            candidate_functions = []
            function_embeddings = []
            candidate_lines = []
            line_embeddings = []
            
            # Process each candidate file
            for file_idx, file_info in enumerate(similar_files):
                file_repo = file_info['repo']
                file_path = file_info['path']
                file_repo_simple = file_repo.split('/')[-1]
                
                # Skip if repo doesn't exist (just in case)
                repo_dir = os.path.join("../data/repos", file_repo_simple)
                if not os.path.exists(repo_dir):
                    print(f"Repository {file_repo} not found at {repo_dir}")
                    continue
                
                # Get SHA for this file
                sha = bug['before_fix_sha'] if file_idx == correct_file_idx else "HEAD"
                
                # Checkout repo to correct SHA
                checkout_repo(repo_dir, sha)
                
                # Get full path to file
                full_path = os.path.join(repo_dir, file_path)
                
                # Skip if file doesn't exist
                if not os.path.exists(full_path):
                    print(f"File not found: {full_path}")
                    continue
                
                # Add to candidate files
                candidate_files.append(file_path)
                
                # Get file embedding
                if file_info['index'] >= 0:
                    # Use embedding from FAISS
                    file_embedding = resources['faiss_index'].reconstruct(file_info['index'])
                else:
                    # Use buggy file embedding
                    file_embedding = resources['buggy_file_embeddings'][
                        resources['bug_id_to_embedding'][bug_id]['file_embedding_index']
                    ]
                
                file_embeddings.append(file_embedding.tolist())
                
                # Target line for this file
                target_line = buggy_line_number if file_idx == correct_file_idx else None
                
                # Extract functions from file
                functions = extract_functions_regex(full_path, target_line)
                for function in functions:
                    # Add to candidate functions
                    candidate_functions.append({
                        'file_idx': len(candidate_files) - 1,
                        'name': function['name'],
                        'class': function['class'],
                        'start_line': function['start_line'],
                        'end_line': function['end_line']
                    })
                    # Generate function embedding
                    function_embedding = embedder.embed_function(function['content'])
                    function_embeddings.append(function_embedding.tolist())


                    # Extract lines from function
                    lines = extract_lines_from_function(function, full_path, target_line)
                    
                    # Process lines
                    for line in lines:
                        line_data = {
                            'function_idx': len(candidate_functions) - 1,
                            'line_num': line['line_num'],
                            'content': line['content']
                        }
                        candidate_lines.append(line_data)
                        
                        # Generate line embedding
                        line_embedding = embedder.embed_line(line['content'])
                        line_embeddings.append(line_embedding.tolist())
            
            # Find correct function and line indices
            correct_function_idx = -1
            correct_line_indices = []
            
            for i, function in enumerate(candidate_functions):
                # Check if function is in the correct file
                if function['file_idx'] == correct_file_idx:
                    # Check if function contains buggy line
                    if buggy_function_name and function['name'] == buggy_function_name:
                        correct_function_idx = i
                    elif function['start_line'] <= buggy_line_number <= function['end_line']:
                        correct_function_idx = i
            
            # Find correct line indices
            for i, line in enumerate(candidate_lines):
                # Check if line is in the correct function
                if correct_function_idx >= 0 and line['function_idx'] == correct_function_idx:
                    # Check if this is the buggy line
                    if line['line_num'] == buggy_line_number:
                        correct_line_indices.append(i)
            
            # Get stack trace embedding
            if bug_id in resources['bug_id_to_embedding']:
                trace_idx = resources['bug_id_to_embedding'][bug_id]['trace_embedding_index']
                stack_trace_embedding = resources['trace_embeddings'][trace_idx].tolist()
            else:
                stack_trace_embedding = np.zeros(768, dtype=np.float32).tolist()
            
            # Create dataset entry
            entry = {
                'stack_trace': bug['stack_trace'],
                'stack_trace_embedding': stack_trace_embedding,
                
                # 'candidate_files': candidate_files,
                # 'candidate_functions': candidate_functions,
                # 'candidate_lines': candidate_lines,

                'file_embeddings': file_embeddings,
                'correct_file_idx': correct_file_idx,
                
                'function_embeddings': function_embeddings,
                'correct_function_idx': correct_function_idx,
                
                'line_embeddings': line_embeddings,
                'correct_line_idx': correct_line_indices,
                
                'metadata': {
                    'repo': repo_name,
                    'bug_id': bug_id,
                    'sha': bug['before_fix_sha'],
                    'buggy_file_path': buggy_file_path,
                    'buggy_line_number': buggy_line_number
                }
            }
            
            dataset.append(entry)
            
        except Exception as e:
            print(f"Error processing bug {bug.get('issue_id', 'unknown')}: {e}")
            traceback.print_exc()
    
    return dataset

def main():
    # Setup paths
    data_dir = Path("../data")
    output_dir = data_dir / "hierarchical_dataset"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load filtered bug dataset
    bug_dataset_path = data_dir / "bug_localization_dataset_filtered.json"
    print(f"Loading bug dataset from {bug_dataset_path}")
    with open(bug_dataset_path, 'r') as f:
        bugs = json.load(f)
    
    print(f"Loaded {len(bugs)} bugs from filtered dataset")
    
    # Load existing resources
    resources = load_resources(data_dir)
    print("Resources loaded successfully")
    
    # Initialize embedder for function and line embeddings
    embedder = CodeEmbedder()
    
    # Split into train/val/test
    # Use repo name for stratification to ensure each split has diverse repositories
    print("Splitting dataset...")
    repos = [bug['repo_name'] for bug in bugs]
    
    # Train 70%, val 15%, test 15%
    train_bugs, temp_bugs = train_test_split(bugs, test_size=0.3, random_state=42, stratify=repos)
    val_bugs, test_bugs = train_test_split(temp_bugs, test_size=0.5, random_state=42)
    
    print(f"Split dataset into {len(train_bugs)} train, {len(val_bugs)} validation, {len(test_bugs)} test bugs")
    
    batch_size = 50  # Process 50 bugs at a time
    
    # Process each split
    print("\nProcessing training set...")
    train_dataset = create_hierarchical_dataset(train_bugs, resources, embedder)
    
    print("\nProcessing validation set...")
    val_dataset = create_hierarchical_dataset(val_bugs, resources, embedder)
    
    print("\nProcessing test set...")
    test_dataset = create_hierarchical_dataset(test_bugs, resources, embedder)
    

    # Save datasets
    print("\nSaving datasets...")
    
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_dataset, f)
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_dataset, f)
    
    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_dataset, f)
    
    print(f"\nProcessed and saved {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test examples")
    print(f"Datasets saved to {output_dir}")

if __name__ == "__main__":
    main()