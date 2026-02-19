import gymnasium as gym
import minigrid
import numpy as np
import json
import os
from collections import deque

# --- Expert Bot for GoToDoor ---
def solve_gotodoor(env):
    """
    Expert policy for GoToDoor.
    Logic: Turn until the door is seen, then move towards it.
    """
    # Find the door
    door = None
    for obj in env.grid.grid:
        if obj and obj.type == 'door':
            door = obj
            break
    if not door: return None # Should not happen

    # Simple BFS to find path
    q = deque([(env.agent_pos, [])])
    visited = {env.agent_pos}
    while q:
        pos, path = q.popleft()
        if pos == door.pos:
            return path
        for action in [env.actions.forward, env.actions.left, env.actions.right]:
            # Simulate action
            env.agent_pos = pos
            if action == env.actions.left:
                env.agent_dir = (env.agent_dir - 1) % 4
            elif action == env.actions.right:
                env.agent_dir = (env.agent_dir + 1) % 4
            
            fwd_pos = env.front_pos
            fwd_cell = env.grid.get(*fwd_pos)

            new_pos = env.agent_pos
            if action == env.actions.forward and (fwd_cell is None or fwd_cell.can_overlap()):
                new_pos = fwd_pos

            if new_pos not in visited:
                visited.add(new_pos)
                q.append((new_pos, path + [action]))
    return None # No path found

# --- Expert Bot for PickupDist ---
def solve_pickup(env):
    """
    Expert policy for PickupDist.
    Logic: Navigate to the object, then pick it up.
    """
    # This is a simplified version. A full BFS is needed for optimal paths.
    # We will use a simple greedy approach.
    
    # Find the object
    target_obj = None
    for obj in env.grid.grid:
        if obj and obj.type == env.target_type and obj.color == env.target_color:
            target_obj = obj
            break
    if not target_obj: return None

    # Simple navigation + pickup
    path_to_obj = solve_gotodoor(env) # Re-use GoTo logic
    if path_to_obj is None: return None
    
    return path_to_obj + [env.actions.pickup]

# --- Expert Bot for Unlock ---
def solve_unlock(env):
    """
    Expert policy for Unlock.
    Logic: Go to key, pickup, go to door, unlock.
    """
    # This is a simplified version.
    # Find key and door
    key = None
    door = None
    for obj in env.grid.grid:
        if obj and obj.type == 'key':
            key = obj
        if obj and obj.type == 'door':
            door = obj
    if not key or not door: return None

    # Path to key
    env.target_pos = key.pos # Temporarily set target for GoTo
    path_to_key = solve_gotodoor(env)
    if path_to_key is None: return None
    
    # Path from key to door
    env.agent_pos = key.pos
    env.target_pos = door.pos
    path_to_door = solve_gotodoor(env)
    if path_to_door is None: return None

    return path_to_key + [env.actions.pickup] + path_to_door + [env.actions.toggle]


# --- Trajectory Generation ---

def grid_to_string(grid, agent_pos, agent_dir):
    """Converts the grid observation to a string."""
    h, w = grid.height, grid.width
    grid_str = ""
    for i in range(h):
        for j in range(w):
            if (j, i) == agent_pos:
                grid_str += ">v<^"[agent_dir]
                continue
            
            cell = grid.get(j, i)
            if cell is None:
                grid_str += "."
            elif cell.type == 'wall':
                grid_str += "#"
            elif cell.type == 'door':
                grid_str += "D"
            elif cell.type == 'key':
                grid_str += "K"
            elif cell.type == 'ball':
                grid_str += "B"
            else:
                grid_str += "?"
        grid_str += "\\n"
    return grid_str

def generate_trajectories(env_name, solver_fn, num_trajectories, output_path):
    """Generates expert trajectories and saves them to a JSONL file."""
    env = gym.make(env_name)
    
    with open(output_path, 'w') as f:
        for i in range(num_trajectories):
            print(f"Generating trajectory {i+1}/{num_trajectories} for {env_name}...")
            obs, info = env.reset()
            
            # Get the full plan from the expert solver
            plan = solver_fn(env)
            if not plan:
                print(f"  Solver failed for this seed. Skipping.")
                continue

            messages = []
            done = False
            truncated = False
            
            for action in plan:
                if done or truncated: break
                
                # Format observation
                obs_str = f"Mission: {obs['mission']}\\n" + grid_to_string(env.grid, env.agent_pos, env.agent_dir)
                messages.append({"role": "user", "content": obs_str})
                
                # Get expert action from the plan
                action_str = env.actions(action).name
                messages.append({"role": "assistant", "content": action_str})
                
                obs, reward, done, truncated, info = env.step(action)

            # Write trajectory to file if it was successful
            if done:
                f.write(json.dumps({"messages": messages}) + "\\n")
            else:
                print(f"  Trajectory did not reach goal. Skipping.")

    env.close()

if __name__ == "__main__":
    output_dir = "any_order_training/data"
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = {
        "minigrid-GoToDoor-v0": solve_gotodoor,
        "minigrid-PickupDist-v0": solve_pickup,
        "minigrid-Unlock-v0": solve_unlock
    }
    
    for env_name, solver in tasks.items():
        output_path = os.path.join(output_dir, f"{env_name}_expert.jsonl")
        generate_trajectories(env_name, solver, num_trajectories=100, output_path=output_path)
        print(f"Saved trajectories to {output_path}")

    print("\n--- Data Generation Complete ---")
    print("You can now run experiments using the generated .jsonl files in the 'any_order_training/data' directory.")

