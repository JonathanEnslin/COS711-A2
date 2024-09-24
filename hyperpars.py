import copy

simple_example = {
    'hp1': [1, 2],
    'optimiser': [
        {"type": 'sgd', "lr": [0.1, 0.01]},
        {"type": 'adam', "beta":
          [
              {"type": "int", "val1": [1, 2], "val2": [4, 5]},
              {"type": "str", "val": ['a', 'b', 'c']},

          ]}],
    # 'hp2': ["a", "b"],
    # 'hp3': ["+", "-"],
}

def recursive_build_search_space(option_space, search_space, current_combination):
    if len(option_space.keys()) == 0:
        search_space.append(current_combination)
        return

    opt_name = list(option_space.keys())[0]
    trimmed_option_space = copy.deepcopy(option_space)
    del trimmed_option_space[opt_name]

    current_option_space = option_space[opt_name]
    if not isinstance(current_option_space, list) and not isinstance(current_option_space, dict):
        current_option_space = [current_option_space]
        
    for opt_val in current_option_space:
        if isinstance(opt_val, dict):
            inner_option_space = opt_val
            inner_search_space = []
            inner_current_combo = {}
            recursive_build_search_space(inner_option_space, inner_search_space, inner_current_combo)
        else:
            inner_search_space = [opt_val]
            
        for opt in inner_search_space:
            combination = copy.deepcopy(current_combination)
            combination[opt_name] = opt
            recursive_build_search_space(trimmed_option_space, search_space, combination)
    


search_space = []
recursive_build_search_space(option_space=simple_example, search_space=search_space, current_combination={})

for combination in search_space:
    print(combination)

print(len(search_space))