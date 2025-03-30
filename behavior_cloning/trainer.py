import torch
from torch.utils.data import DataLoader, Dataset
from collections import deque
import random
import interactive_interface as ii
from envs.unity_environment import UnityEnvironment
import bc_agent
from envs.utils.utils_environment import check_progress, check_progress_action_put, check_progress_action_open, check_progress_action_grab

class ADGTrainer:
    def __init__(self, args, model: bc_agent, vh_env: UnityEnvironment):
        self.model = model
        self.vh_env = vh_env
        self.comm = vh_env.comm
        self.args = args
        self.replay_buffer = deque(maxlen=args.buffer_size)
        # 扩展词汇表以支持多种谓词
        self.action_vocab = {"Grab": 0, "Put": 1, "Open": 2, "Close": 3}
        self.obj_vocab = {"apple": 0, "fridge": 1, "table": 2, "door": 3}
        self.state_vocab = {"ON": 0, "INSIDE": 1, "OPEN": 2, "CLOSED": 3}  # 状态/谓词词汇表
        self.goal_set = self.initialize_goal_set()

    # TODO: 没啥用，肯定得重写
    def initialize_goal_set(self):
        # 初始化目标集，支持多种谓词
        self.comm.reset(0)
        _, graph = self.vh_env.get_graph()
        goals = []
        for node in graph["nodes"]:
            obj_name = node["name"]
            obj_id = node["id"]
            if obj_name in self.obj_vocab:
                # ON 和 INSIDE 目标
                for loc_node in graph["nodes"]:
                    if loc_node["id"] != obj_id:
                        goals.append({f"ON_{obj_name}_{loc_node['id']}": 1})
                        goals.append({f"INSIDE_{obj_name}_{loc_node['id']}": 1})
                # OPEN 和 CLOSE 目标
                if "states" in node and ("OPEN" in node["states"] or "CLOSED" in node["states"]):
                    goals.append({f"OPEN_{obj_name}_{obj_id}": 1})
                    goals.append({f"CLOSE_{obj_name}_{obj_id}": 1})
        return goals[:10]  # 限制初始数量

    def explore(self, num_episodes):
        trajectories = []
        for _ in range(num_episodes):
            goal, initial_state = self.sample_goal_and_state()
            trajectory = self.collect_trajectory(initial_state, goal)
            relabeled_trajectory = self.relabel_trajectory(trajectory)
            trajectories.append(relabeled_trajectory)
            self.goal_set.append(relabeled_trajectory["goal"])
        return trajectories

    def collect_trajectory(self, initial_state, goal):
        # self.comm.reset(0)
        self.vh_env.raw_reset()
        self.comm.add_character()
        obs, actions = [initial_state], []
        state = initial_state
        for _ in range(self.args.max_steps):
            action = self.get_action(state, goal)
            success, _ = self.comm.render_script(script=[action], recording=False)
            if not success:
                break
            new_state = self.vh_env.get_graph()
            obs.append(new_state)
            actions.append(action)
            state = new_state
        return {"goal": goal, "obs": obs, "actions": actions}

    def get_action(self, state, goal): # TODO: params需修改
        input_data = self.format_input(state, goal, [])  # TODO: 需实现
        action_logits, obj_logits = self.model(input_data)
        # TODO: 手动适配用法
        action_str, _, _ = ii.sample_action(
            args=self.args,
            obs=[state],  # 假设 obs 是列表
            agent_id=0,
            action_logits=action_logits,
            object_logits=obj_logits,
            all_actions=[],
            all_cur_observation=[],
            logging=None
        )
        return action_str
    
    def train(self):
        for iteration in range(self.args.num_iterations):
            trajectories = self.explore(self.args.max_episode_length)
            for traj in trajectories:
                self.replay_buffer.append(self.format_trajectory(traj))
            dataset = ADGDataset(list(self.replay_buffer))
            trainloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            output = self.model.run(trainloader, iteration, mode='train')
            print(f"Iteration {iteration}: Loss={output[0]}, Top1={output[3]}")

    def sample_goal_and_state(self):
        goal = random.choice(self.goal_set)
        self.comm.reset(0)
        _, initial_state = self.comm.get_graph()
        return goal, initial_state

    def relabel_trajectory(self, trajectory):
        # obs = trajectory["obs"]
        # actions = trajectory["actions"]
        # final_state = obs[-1]
        # original_goal = trajectory["goal"]

        # # 检查原始目标
        # satisfied, unsatisfied = check_progress(final_state, original_goal)
        # if all(v == 0 for v in unsatisfied.values()):
        #     return trajectory

        # # 从最终状态和动作推断新目标
        # new_goal = {}
        # id2node = {node["id"]: node for node in final_state["nodes"]}

        # # 优先从动作推断
        # for action in reversed(actions):
        #     if "Put" in action:
        #         obj = action.split()[1].strip("<>")
        #         loc = action.split()[2].strip("<>")
        #         obj_id = next((n["id"] for n in final_state["nodes"] if n["name"] == obj), None)
        #         loc_id = next((n["id"] for n in final_state["nodes"] if n["name"] == loc), None)
        #         if obj_id and loc_id:
        #             for edge in final_state["edges"]:
        #                 if edge["from_id"] == obj_id and edge["to_id"] == loc_id:
        #                     rel_type = edge["relation_type"].upper()
        #                     new_goal[f"{rel_type}_{obj}_{loc_id}"] = 1
        #                     return {"goal": new_goal, "obs": obs, "actions": actions}
        #     elif "Open" in action:
        #         obj = action.split()[1].strip("<>")
        #         obj_id = next((n["id"] for n in final_state["nodes"] if n["name"] == obj), None)
        #         if obj_id and "OPEN" in id2node[obj_id]["states"]:
        #             new_goal[f"OPEN_{obj}_{obj_id}"] = 1
        #             return {"goal": new_goal, "obs": obs, "actions": actions}

        # # 如果动作无明确目标，从状态推断
        # for edge in final_state["edges"]:
        #     rel_type = edge["relation_type"].upper()
        #     if rel_type in ["ON", "INSIDE"]:
        #         obj_name = id2node[edge["from_id"]]["name"]
        #         loc_id = edge["to_id"]
        #         if obj_name in self.obj_vocab:
        #             new_goal[f"{rel_type}_{obj_name}_{loc_id}"] = 1
        #             return {"goal": new_goal, "obs": obs, "actions": actions}
        # for node in final_state["nodes"]:
        #     if "states" in node and node["name"] in self.obj_vocab:
        #         if "OPEN" in node["states"]:
        #             new_goal[f"OPEN_{node['name']}_{node['id']}"] = 1
        #             return {"goal": new_goal, "obs": obs, "actions": actions}
        #         elif "CLOSED" in node["states"]:
        #             new_goal[f"CLOSE_{node['name']}_{node['id']}"] = 1
        #             return {"goal": new_goal, "obs": obs, "actions": actions}

        # return trajectory  # 默认返回原轨迹
        obs = trajectory["obs"]
        actions = trajectory["actions"]
        init_graph = obs[0]  # 初始状态
        cur_graph = obs[-1]  # 最终状态
        original_goal = trajectory["goal"]

        # 检查原始目标是否满足
        satisfied, unsatisfied = check_progress(cur_graph, original_goal)
        if all(v == 0 for v in unsatisfied.values()):
            return trajectory  # 原目标已满足

        # 从状态变化和动作推断新目标
        new_goal = {}
        id2node = {node["id"]: node for node in cur_graph["nodes"]}

        # 优先从动作推断
        for action in reversed(actions):
            if "[Put" in action:
                obj = action.split()[1].strip("<>")
                tar = action.split()[2].strip("<>")
                obj_tar_edges = check_progress_action_put(cur_graph, obj, tar, "put " + obj + " on " + tar)
                init_edges = check_progress_action_put(init_graph, obj, tar, "put " + obj + " on " + tar)
                new_edges = [e for e in obj_tar_edges if e not in init_edges]
                if new_edges:
                    rel_type = new_edges[0]["relation_type"].upper()
                    obj_id = new_edges[0]["from_id"]
                    tar_id = new_edges[0]["to_id"]
                    new_goal[f"{rel_type}_{obj}_{tar_id}"] = 1
                    return {"goal": new_goal, "obs": obs, "actions": actions}
            elif "[Open" in action:
                obj = action.split()[1].strip("<>")
                open_nodes = check_progress_action_open(cur_graph, obj, "open " + obj)
                init_open = check_progress_action_open(init_graph, obj, "open " + obj)
                new_open = [n for n in open_nodes if n not in init_open]
                if new_open:
                    obj_id = new_open[0]["id"]
                    new_goal[f"OPEN_{obj}_{obj_id}"] = 1
                    return {"goal": new_goal, "obs": obs, "actions": actions}
            elif "[Grab" in action:
                obj = action.split()[1].strip("<>")
                grab_edges = check_progress_action_grab(cur_graph, obj, "grab " + obj)
                init_grab = check_progress_action_grab(init_graph, obj, "grab " + obj)
                new_grab = [e for e in grab_edges if e not in init_grab]
                if new_grab:
                    obj_id = new_grab[0]["to_id"] if new_grab[0]["from_id"] == 1 else new_grab[0]["from_id"]
                    new_goal[f"HOLDS_{obj}_{1}"] = 1  # 假设 agent_id=1
                    return {"goal": new_goal, "obs": obs, "actions": actions}

        # 如果动作无明确目标，从状态推断
        for edge in cur_graph["edges"]:
            rel_type = edge["relation_type"].upper()
            if rel_type in ["ON", "INSIDE"] and edge not in init_graph["edges"]:
                obj_name = id2node[edge["from_id"]]["name"]
                tar_id = edge["to_id"]
                new_goal[f"{rel_type}_{obj_name}_{tar_id}"] = 1
                return {"goal": new_goal, "obs": obs, "actions": actions}
        for node in cur_graph["nodes"]:
            if "states" in node and node not in init_graph["nodes"]:
                if "OPEN" in node["states"]:
                    new_goal[f"OPEN_{node['name']}_{node['id']}"] = 1
                    return {"goal": new_goal, "obs": obs, "actions": actions}

        return trajectory  # 默认返回原轨迹

    # def check_progress(self, state, task_goal):
    #     unsatisfied = {}
    #     satisfied = {}
    #     id2node = {node['id']: node for node in state['nodes']}

    #     if not task_goal:
    #         return {}, {}

    #     for key, count in task_goal.items():
    #         elements = key.split('_')
    #         predicate = elements[0].upper()
    #         obj = elements[1]
    #         target_id = int(elements[2])
    #         unsatisfied[key] = count
    #         satisfied[key] = []

    #         if predicate in ["ON", "INSIDE"]:
    #             for edge in state['edges']:
    #                 if (edge['relation_type'].upper() == predicate and
    #                     edge['to_id'] == target_id and
    #                     (id2node[edge['from_id']]['name'] == obj or str(edge['from_id']) == obj)):
    #                     satisfied[key].append(f"{predicate}_{edge['from_id']}_{target_id}")
    #                     unsatisfied[key] -= 1
    #         elif predicate in ["OPEN", "CLOSE"]:
    #             for node in state['nodes']:
    #                 if node["id"] == target_id and node["name"] == obj:
    #                     if (predicate == "OPEN" and "OPEN" in node["states"]) or \
    #                     (predicate == "CLOSE" and "CLOSED" in node["states"]):
    #                         satisfied[key].append(f"{predicate}_{obj}_{target_id}")
    #                         unsatisfied[key] -= 1
    #     return satisfied, unsatisfied