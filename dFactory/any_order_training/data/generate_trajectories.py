
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import numpy as np
import json
import os

def grid_to_string(image, direction):
    """Converts the grid observation to a string."""
    h, w, c = image.shape
    grid_str = ""
    for i in range(h):
        for j in range(w):
            obj_type, color, state = image[i, j]
            if obj_type == 0: # Unseen
                grid_str += "U"
            elif obj_type == 1: # Empty
                grid_str += "."
            elif obj_type == 2: # Wall
                grid_str += "#"
            elif obj_type == 3: # Floor
                grid_str += "_"
            elif obj_type == 4: # Door
                grid_str += "D"
            elif obj_type == 5: # Key
                grid_str += "K"
            elif obj_type == 6: # Ball
                grid_str += "B"
            elif obj_type == 7: # Box
                grid_str += "X"
            elif obj_type == 8: # Goal
                grid_str += "G"
            else:
                grid_str += "?"
        grid_str += "\\n"
    
    dir_str = ["right", "down", "left", "up"][direction]
    
    return f"Direction: {dir_str}\\n{grid_str}"


def generate_trajectories(env_name, num_trajectories, output_path):
    """Generates trajectories from a BabyAI environment and saves them to a JSONL file."""
    env = gym.make(env_name)
    # The 'image' observation is a grid of objects.
    # We don't need ImgObsWrapper as we are not using pixel observations.
    
    with open(output_path, 'w') as f:
        for i in range(num_trajectories):
            print(f"Generating trajectory {i+1}/{num_trajectories} for {env_name}...")
            obs, info = env.reset()
            done = False
            truncated = False
            
            messages = []
            
            while not done and not truncated:
                obs_str = f"Mission: {obs['mission']}\\n" + grid_to_string(obs['image'], obs['direction'])
                messages.append({"role": "user", "content": obs_str})
                
                # Get expert action
                action = env.get_bot_action()
                action_str = env.actions(action).name
                messages.append({"role": "assistant", "content": action_str})
                
                obs, reward, done, truncated, info = env.step(action)

            # Write trajectory to file
            f.write(json.dumps({"messages": messages}) + "\\n")
            
    env.close()


if __name__ == "__main__":
    output_dir = "dFactory/any_order_training/data"
    os.makedirs(output_dir, exist_ok=True)
    
    env_list = [
        "BabyAI-GoToRedBall-v0",
        "BabyAI-GoToLocal-v0",
        "BabyAI-Open-v0",
    ]
    
    for env_name in env_list:
        output_path = os.path.join(output_dir, f"{env_name.lower()}_trajectories.jsonl")
        generate_trajectories(env_name, num_trajectories=10, output_path=output_path)
        print(f"Saved trajectories to {output_path}")
