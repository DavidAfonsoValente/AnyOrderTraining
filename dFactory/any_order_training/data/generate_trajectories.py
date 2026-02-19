
import gymnasium as gym
import minigrid
import numpy as np
import json
import os
from collections import deque

def solve_gotodoor(env):
    """Expert policy for GoToDoor."""
    unwrapped_env = env.unwrapped
    door = None
    for obj in unwrapped_env.grid.grid:
        if obj and obj.type == 'door':
            door = obj
            break
    if not door: return None

    q = deque([(unwrapped_env.agent_pos, unwrapped_env.agent_dir, [])])
    visited = {(unwrapped_env.agent_pos, unwrapped_env.agent_dir)}

    while q:
        pos, direction, path = q.popleft()
        if tuple(pos) == tuple(door.pos):
            return path

        # Try turning
        for turn_action in [env.actions.left, env.actions.right]:
            new_dir = (direction + (1 if turn_action == env.actions.right else -1)) % 4
            if (tuple(pos), new_dir) not in visited:
                visited.add((tuple(pos), new_dir))
                q.append((pos, new_dir, path + [turn_action]))

        # Try moving forward
        fwd_pos = pos + unwrapped_env.dir_vec
        fwd_cell = unwrapped_env.grid.get(*fwd_pos)
        if (fwd_cell is None or fwd_cell.can_overlap()) and (tuple(fwd_pos), direction) not in visited:
            visited.add((tuple(fwd_pos), direction))
            q.append((fwd_pos, direction, path + [env.actions.forward]))
    
    return None

def solve_pickup(env):
    """Expert policy for PickupDist."""
    unwrapped_env = env.unwrapped
    target_obj = None
    for obj in unwrapped_env.grid.grid:
        if obj and obj.type == unwrapped_env.target_type and obj.color == unwrapped_env.target_color:
            target_obj = obj
            break
    if not target_obj: return None
    
    # Create a temporary env to solve for path
    temp_env = gym.make(env.spec.id, render_mode='rgb_array')
    temp_env.reset(seed=env.unwrapped.seed)
    temp_env.unwrapped.agent_pos = unwrapped_env.agent_pos
    temp_env.unwrapped.agent_dir = unwrapped_env.agent_dir
    temp_env.unwrapped.grid = unwrapped_env.grid
    
    path_to_obj = solve_gotodoor(temp_env)
    if path_to_obj is None: return None
    
    return path_to_obj + [env.actions.pickup]


def solve_unlock(env):
    """Expert policy for Unlock."""
    unwrapped_env = env.unwrapped
    key, door = None, None
    for obj in unwrapped_env.grid.grid:
        if obj and obj.type == 'key': key = obj
        if obj and obj.type == 'door': door = obj
    if not key or not door: return None

    # Path to key
    temp_env = gym.make(env.spec.id, render_mode='rgb_array')
    temp_env.reset(seed=env.unwrapped.seed)
    temp_env.unwrapped.agent_pos = unwrapped_env.agent_pos
    temp_env.unwrapped.agent_dir = unwrapped_env.agent_dir
    temp_env.unwrapped.grid = unwrapped_env.grid
    
    path_to_key = solve_gotodoor(temp_env)
    if path_to_key is None: return None
    
    # Simulate path to key to get agent state
    for action in path_to_key:
        temp_env.step(action)
    
    # Path from key to door
    path_to_door = solve_gotodoor(temp_env)
    if path_to_door is None: return None

    return path_to_key + [env.actions.pickup] + path_to_door + [env.actions.toggle]


def grid_to_string(grid, agent_pos, agent_dir):
    """Converts the grid observation to a string."""
    h, w = grid.height, grid.width
    grid_str = ""
    for i in range(h):
        for j in range(w):
            if (j, i) == tuple(agent_pos):
                grid_str += ">v<^"[agent_dir]
                continue
            cell = grid.get(j, i)
            if cell is None: grid_str += "."
            elif cell.type == 'wall': grid_str += "#"
            elif cell.type == 'door': grid_str += "D"
            elif cell.type == 'key': grid_str += "K"
            else: grid_str += "?"
        grid_str += "\\n"
    return grid_str

def generate_trajectories(env_name, solver_fn, num_trajectories, output_path):
    """Generates expert trajectories and saves them to a JSONL file."""
    env = gym.make(env_name, render_mode='rgb_array')
    
    with open(output_path, 'w') as f:
        trajectories_saved = 0
        while trajectories_saved < num_trajectories:
            print(f"Generating trajectory {trajectories_saved + 1}/{num_trajectories} for {env_name}...")
            obs, info = env.reset()
            
            plan = solver_fn(env)
            if not plan:
                print("  Solver failed for this seed. Retrying with a new seed.")
                continue

            messages = []
            done = False
            truncated = False
            
            for action in plan:
                if done or truncated: break
                
                obs_str = f"Mission: {obs['mission']}\\n" + grid_to_string(env.unwrapped.grid, env.unwrapped.agent_pos, env.unwrapped.agent_dir)
                messages.append({"role": "user", "content": obs_str})
                
                action_str = env.actions(action).name
                messages.append({"role": "assistant", "content": action_str})
                
                obs, reward, done, truncated, info = env.step(action)

            if done:
                f.write(json.dumps({"messages": messages}) + "\\n")
                trajectories_saved += 1
            else:
                print("  Trajectory did not reach goal. Retrying with a new seed.")

    env.close()

if __name__ == "__main__":
    output_dir = "any_order_training/data"
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = {
        "MiniGrid-GoToDoor-6x6-v0": solve_gotodoor,
    }
    
    for env_name, solver in tasks.items():
        output_path = os.path.join(output_dir, f"{env_name}_expert.jsonl")
        generate_trajectories(env_name, solver, num_trajectories=100, output_path=output_path)
        print(f"Saved trajectories to {output_path}")

    print("\n--- Data Generation Complete ---")

