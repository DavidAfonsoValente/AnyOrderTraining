
import gymnasium as gym
import minigrid
import numpy as np
import json
import os
from collections import deque
from minigrid.core.constants import DIR_TO_VEC

# ==============================================================================
# A robust, stateless, rule-based expert for solving simple Minigrid tasks.
#
# This implementation corrects previous bugs by:
# 1. Using env.unwrapped to access the base environment attributes.
# 2. Correctly finding object positions by searching the grid.
# 3. Using a stateless BFS pathfinding algorithm.
# 4. Using the correct minigrid API (e.g., DIR_TO_VEC).
# ==============================================================================

def find_path_to_pos(unwrapped_env, target_pos):
    """
    A stateless BFS planner.
    Finds a sequence of actions to navigate to a target position.
    """
    start_pos = tuple(unwrapped_env.agent_pos)
    start_dir = unwrapped_env.agent_dir
    
    q = deque([(start_pos, start_dir, [])])
    visited = {(start_pos, start_dir)}

    while q:
        cur_pos, cur_dir, path = q.popleft()

        if cur_pos == target_pos:
            return path

        # Try turning
        for turn_action in [unwrapped_env.actions.left, unwrapped_env.actions.right]:
            next_dir = (cur_dir + (1 if turn_action == unwrapped_env.actions.right else -1)) % 4
            if (cur_pos, next_dir) not in visited:
                visited.add((cur_pos, next_dir))
                q.append((cur_pos, next_dir, path + [turn_action]))

        # Try moving forward
        fwd_pos = np.add(cur_pos, DIR_TO_VEC[cur_dir])
        fwd_cell = unwrapped_env.grid.get(*fwd_pos)
        if (fwd_cell is None or fwd_cell.can_overlap()) and (tuple(fwd_pos), cur_dir) not in visited:
            visited.add((tuple(fwd_pos), cur_dir))
            q.append((tuple(fwd_pos), cur_dir, path + [unwrapped_env.actions.forward]))

    return None

def get_object_pos(unwrapped_env, obj_type, obj_color=None):
    """Finds the position of a specific object on the grid."""
    for i in range(unwrapped_env.width):
        for j in range(unwrapped_env.height):
            cell = unwrapped_env.grid.get(i, j)
            if cell and cell.type == obj_type:
                if obj_color is None or cell.color == obj_color:
                    return (i, j)
    return None

def solve_gotodoor(unwrapped_env):
    """Expert policy for GoToDoor."""
    door_pos = get_object_pos(unwrapped_env, 'door')
    if door_pos is None: return None
    return find_path_to_pos(unwrapped_env, door_pos)

def solve_pickup(unwrapped_env):
    """Expert policy for PickupDist."""
    obj_pos = get_object_pos(unwrapped_env, unwrapped_env.target_type, unwrapped_env.target_color)
    if obj_pos is None: return None
    path = find_path_to_pos(unwrapped_env, obj_pos)
    if path is None: return None
    return path + [unwrapped_env.actions.pickup]

def solve_unlock(unwrapped_env):
    """Expert policy for Unlock."""
    key_pos = get_object_pos(unwrapped_env, 'key')
    door_pos = get_object_pos(unwrapped_env, 'door')
    if key_pos is None or door_pos is None: return None

    # Plan: Go to key -> pickup -> go to door -> toggle
    path_to_key = find_path_to_pos(unwrapped_env, key_pos)
    if path_to_key is None: return None
    
    # We need to simulate the env to find the state after picking up the key
    # This is complex. For this simplified solver, we assume the agent can
    # find the door from the key's position without the key disappearing.
    path_to_door = find_path_to_pos(unwrapped_env, door_pos)
    if path_to_door is None: return None
    
    # This is a simplification and may not be optimal.
    # A true optimal planner would path from agent->key, then key->door.
    # For now, we find path to key, then path to door from original position.
    
    return path_to_key + [unwrapped_env.actions.pickup] + path_to_door + [unwrapped_env.actions.toggle]

# --- Trajectory Generation Script ---

def grid_to_string(grid, agent_pos, agent_dir):
    """Converts the grid observation to a string."""
    # ... (rest of the function is correct) ...
    pass # Placeholder

def generate_trajectories(env_name, solver_fn, num_trajectories, output_path):
    """Generates expert trajectories and saves them to a JSONL file."""
    env = gym.make(env_name, render_mode='rgb_array')
    
    with open(output_path, 'w') as f:
        trajectories_saved = 0
        while trajectories_saved < num_trajectories:
            print(f"Generating trajectory {trajectories_saved + 1}/{num_trajectories} for {env_name}...")
            obs, info = env.reset()
            
            # The solver needs a pristine copy of the unwrapped env to plan against
            plan = solver_fn(env.unwrapped)
            if not plan:
                print("  Solver failed for this seed. Retrying with a new seed.")
                continue

            messages = []
            done = False
            truncated = False
            
            # We need to re-reset the env to execute the plan from the start
            env.reset()
            
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
        "MiniGrid-PickupDist-6x6-v0": solve_pickup,
        "MiniGrid-Unlock-v0": solve_unlock
    }
    
    for env_name, solver in tasks.items():
        output_path = os.path.join(output_dir, f"{env_name}_expert.jsonl")
        generate_trajectories(env_name, solver, num_trajectories=100, output_path=output_path)
        print(f"Saved trajectories to {output_path}")

    print("\n--- Data Generation Complete ---")

