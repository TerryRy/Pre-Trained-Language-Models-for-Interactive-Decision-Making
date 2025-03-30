def convert_actions_agent(actions, agent_id=0):
    '''
    [
    "[WALK] <home_office>",
    "[WALK] <couch>"
    ]

    to

    [
    {0, "[WALK] <home_office>"},
    {0, "[WALK] <couch>"}
    ]
    '''
    converted = []
    for i in range(0, len(actions)):
        action_dict = {agent_id : actions[i]}
        converted.append(action_dict)
    return converted


import pickle
import os
import sys

file_path = os.path.join('../data/test_init_env', 'InDistributation.p')
with open(file_path, "rb") as f:
    data_info = pickle.load(f)

# print(type(data_info))  # 查看数据类型
# print(data_info[0].keys())  # 如果是字典，打印键
# print(data_info['max_node_class_name_gpt2_length'])
data_example = data_info[0]
for key in data_example.keys():
    if isinstance(data_example[key], dict):
        if key == "task_goal":
            val = f"{data_example[key][0]} # {data_example[key][1]}"
        else:
            # val = data_example[key].keys()
            val = data_example[key]["nodes"][0].keys()
    elif isinstance(data_example[key], list):
        val = data_example[key][0]
    elif isinstance(data_example[key], int) or isinstance(data_example[key], str):
        val = data_example[key]
    else:
        val = data_example[key]
        # val = 'ERROR'
    curr = f"{type(data_example[key])}"
    print(f"name:  {key:<18s}  type:  {curr:<20s} sample:  {val}")

# print(type(data_example["init_graph"]["id"]))
for item in data_example["init_graph"]["nodes"]:
    if item["id"] == 123:
        print(item["category"])
        print(item["class_name"])
        print(item["prefab_name"])



