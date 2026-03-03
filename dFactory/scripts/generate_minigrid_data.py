import json
import os
import argparse
import collections
import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType
import gym_minigrid.wrappers
from gym_minigrid.minigrid import MiniGridEnv, Grid, WorldObj
from tqdm import tqdm

# --- Textual Observation Functions ---

def describe_obj(obj: WorldObj) -> str:
    """Creates a textual description of a single MiniGrid object."""
    if obj is None:
        return "empty"
    return f"{obj.color} {obj.type}"

def describe_obs(obs: ObsType, env: MiniGridEnv) -> str:
    """Creates a textual description of a MiniGrid observation."""
    grid: Grid = Grid.decode(obs['image'])[0]
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir

    # Describe what's in front
    front_pos = agent_pos + env.dir_vec
    front_cell = grid.get(*front_pos)
    front_desc = f"In front of you is {describe_obj(front_cell)}."

    # Describe what's being carried
    carrying_desc = "You are carrying nothing."
    if env.carrying:
        carrying_desc = f"You are carrying a {describe_obj(env.carrying)}."

    # Simple description of left and right cells
    left_vec = np.array((-env.dir_vec[1], env.dir_vec[0]))
    right_vec = np.array((env.dir_vec[1], -env.dir_vec[0]))
    left_pos = agent_pos + left_vec
    right_pos = agent_pos + right_vec
    left_cell = grid.get(*left_pos)
    right_cell = grid.get(*right_pos)
    
    side_desc = f"To your left is {describe_obj(left_cell)}, to your right is {describe_obj(right_cell)}."
    
    return f"{front_desc} {side_desc} {carrying_desc}"


# --- BFS Expert Agent ---

class BFSAgent:
    """An expert agent that uses Breadth-First Search to find paths."""
    ACTION_MAP = {
        0: 'left',
        1: 'right',
        2: 'forward'
    }
    def __init__(self, grid: Grid, agent_pos: tuple, agent_dir: int):
        self.grid = grid
        self.height = grid.height
        self.width = grid.width
        self.initial_pos = tuple(agent_pos)
        self.initial_dir = agent_dir

    def get_path(self, goal_pos: tuple) -> list[ActType]:
        q = collections.deque([((self.initial_pos, self.initial_dir), [])])
        visited = {(self.initial_pos, self.initial_dir)}

        while q:
            (pos, direction), path = q.popleft()
            
            if pos == goal_pos:
                return path

            # Try turning left
            new_dir_left = (direction - 1) % 4
            if ((pos, new_dir_left)) not in visited:
                visited.add((pos, new_dir_left))
                q.append(((pos, new_dir_left), path + [MiniGridEnv.Actions.left]))

            # Try turning right
            new_dir_right = (direction + 1) % 4
            if ((pos, new_dir_right)) not in visited:
                visited.add((pos, new_dir_right))
                q.append(((pos, new_dir_right), path + [MiniGridEnv.Actions.right]))
            
            # Try moving forward
            fwd_pos_map = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
            fwd_vec = fwd_pos_map[direction]
            next_pos = (pos[0] + fwd_vec[0], pos[1] + fwd_vec[1])
            
            if 0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height:
                cell = self.grid.get(*next_pos)
                if (cell is None or cell.can_overlap()) and ((next_pos, direction)) not in visited:
                    visited.add(((next_pos, direction),))
                    q.append(((next_pos, direction), path + [MiniGridEnv.Actions.forward]))
        
        return [] # No path found

# --- Main Generation Logic ---

def generate_trajectories(env_id, num_trajectories, output_file, seed):
    print(f"Generating {num_trajectories} expert trajectories for {env_id}...")
    env = gym.make(env_id)
    
    with open(output_file, 'w') as f:
        for i in tqdm(range(num_trajectories), desc=f"Generating {env_id}"):
            # For reproducibility of envs
            obs, _ = env.reset(seed=seed + i)
            
            trajectory = []
            
            # --- Generate High-Level Plan ---
            # This is a simplified expert for the three target environments
            goals = []
            if "Unlock" in env_id:
                # Find key -> go to key -> pickup -> find door -> go to door -> unlock
                key_pos, key_obj = env.grid.find_obj(WorldObj('key'))
                door_pos, door_obj = env.grid.find_obj(WorldObj('door'))
                goals = [(key_pos, "pickup"), (door_pos, "toggle")]
            elif "Pickup" in env_id:
                # Find ball -> go to ball -> pickup
                ball_pos, _ = env.grid.find_obj(WorldObj('ball'))
                goals = [(ball_pos, "pickup")]
            elif "GoTo" in env_id:
                # Find door -> go to door
                door_pos, _ = env.grid.find_obj(WorldObj('door'))
                goals = [(door_pos, None)]
            
            # --- Execute Plan ---
            done = False
            for goal_pos, interaction in goals:
                if done: break
                
                # 1. Pathfind to the goal position
                agent = BFSAgent(env.grid, env.agent_pos, env.agent_dir)
                action_plan = agent.get_path(goal_pos)

                if not action_plan:
                    # Could not find a path, this trajectory is a failure, skip it
                    print(f"Warning: Could not find path in episode {i+1}. Skipping.")
                    break
                
                # 2. Execute navigation actions
                for action in action_plan:
                    if done: break
                    text_obs = describe_obs(obs, env)
                    action_str = BFSAgent.ACTION_MAP.get(action, 'unknown')
                    trajectory.append({"observation": text_obs})
                    trajectory.append({"action": action_str})
                    obs, _, done, _, _ = env.step(action)
                
                # 3. Execute interaction
                if not done and interaction:
                    text_obs = describe_obs(obs, env)
                    action_enum = getattr(MiniGridEnv.Actions, interaction)
                    trajectory.append({"observation": text_obs})
                    trajectory.append({"action": interaction})
                    obs, _, done, _, _ = env.step(action_enum)
            
            # Add final observation if episode ended
            if trajectory:
                text_obs = describe_obs(obs, env)
                trajectory.append({"observation": text_obs})
                f.write(json.dumps({"trajectory": trajectory}) + '
')
    
    print(f"Successfully generated {num_trajectories} trajectories and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert trajectories for MiniGrid environments.")
    parser.add_argument("--env-id", type=str, required=True, help="The MiniGrid environment ID (e.g., 'MiniGrid-Unlock-v0').")
    parser.add_argument("--num-trajectories", type=int, default=1000, help="Number of trajectories to generate.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment generation.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    generate_trajectories(args.env_id, args.num_trajectories, args.output_file, args.seed)
