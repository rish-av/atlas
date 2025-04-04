import json 

with open('/Users/rishavsinha/Documents/atlas/dataset/test.json') as f:
    data = json.load(f)

print(data[0].keys())

#dict_keys(['stack_trace', 
# 'stack_trace_embedding', 
# 'file_embeddings', 'correct_file_idx', 
# 'function_embeddings', 'correct_function_idx', 'line_embeddings', 
# 'correct_line_idx', 'metadata'])