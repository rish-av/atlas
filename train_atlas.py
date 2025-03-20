from data_loader import CodeBERTEmbedder, BugLocalizationEnv
from atlas_agent import AtlasAgent

def main():
    embedder = CodeBERTEmbedder()
    env = BugLocalizationEnv(embedder)
    file_state_dim = 768 + 64 + 128
    function_state_dim = 768 + 128
    line_state_dim = 768 + 128
    num_files = len(env.project)
    num_functions = len(env.project[0]["functions"])
    num_lines = len(env.project[0]["functions"][0]["lines"])
    agent = AtlasAgent(file_state_dim, function_state_dim, line_state_dim,
                       num_files, num_functions, num_lines)
    print("Starting training...")
    agent.train(env, episodes=500, update_target_every=10, batch_size=32)
    print("Training completed.")
    print("Evaluating agent performance...")
    metrics = agent.evaluate(env, num_episodes=100)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
