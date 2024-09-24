import copy

# simple_example = {
#     'lr': [0.1, 0.01],
#     'optimizer': {
#         'sgd': {
#             'momentum': [0.7, 0.8],
#         },
#         'adam': {
#             'beta1': {
#                 "c1": ['a', 'b'],
#                 "c2": ['+', '-']
#             },
#             'beta2': [0.3, 0.4],
#         },
#     },
# }

simple_example = {
    'optimizer': {
        'sgd': {
            'momentum': {
                "what": [1, 2]
            },
        },
    },
}

def recursive_build_search_space(option_space, search_space, current_combination={}):
    if len(option_space.keys()) == 0:
        search_space.append(current_combination)
        return

    opt_name = list(option_space.keys())[0]
    trimmed_option_space = copy.deepcopy(option_space)
    del trimmed_option_space[opt_name]

    current_option_space = option_space[opt_name]

    if isinstance(current_option_space, dict):
        for choice in current_option_space.keys():
            choice_search_space = []
            choice_option_space = current_option_space[choice]
            recursive_build_search_space(choice_option_space, choice_search_space)
            for choice_opt_val in choice_search_space:
                combination = copy.deepcopy(current_combination)
                combination[opt_name] = {choice: choice_opt_val}
                recursive_build_search_space(trimmed_option_space, search_space, combination)
    else:
        for opt_val in current_option_space:
            combination = copy.deepcopy(current_combination)
            combination[opt_name] = opt_val
            recursive_build_search_space(trimmed_option_space, search_space, combination)

def build_search_space(option_space):
    search_space = []
    recursive_build_search_space(option_space=option_space, search_space=search_space)
    return search_space

search_space = []
recursive_build_search_space(option_space=simple_example, search_space=search_space)

for combination in search_space:
    print(combination)

print(len(search_space))