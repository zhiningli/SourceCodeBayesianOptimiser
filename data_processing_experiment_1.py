from src.data.db.script_crud import ScriptRepository


repo = ScriptRepository()


script_objects = repo.fetch_all_scripts()

original_configuration_counts = 100 * 50 * 6 * 18

number_of_configurations = {}

# Compute the average reduction in search space
for i, script_object in enumerate(script_objects):

    constrained_search_space = script_object["constrained_search_space"]["search_space"]

    lower = constrained_search_space["lower"]
    upper = constrained_search_space["upper"]
    configuration_count = 1
    for j in range(len(lower)):
        configuration_count *= upper[j] - lower[j] 
    number_of_configurations[i] = configuration_count / original_configuration_counts

average = sum(number_of_configurations.values()) / 100

print(f"The constrained search space is on average: {average * 100:.2f}% of the original search space")


# Compute the number of performance

improved_performance_count = 0
same_performance_count = 0
full_percentage_count = 0
decreased_performance_count = 0

improved = {}
decreased= {}
same_best_score = []
# Compute how many scripts faced an increase in accuracy
for i, script_object in enumerate(script_objects):

    score_under_full_search_space = script_object["best_score"]
    score_under_constrained_search_space = script_object["constrained_search_space"]["best_score"]

    if score_under_full_search_space < score_under_constrained_search_space:
        improved_performance_count += 1
        improved[i] = score_under_constrained_search_space - score_under_full_search_space
    elif score_under_constrained_search_space < score_under_full_search_space:
        decreased_performance_count += 1
        decreased[i] = score_under_constrained_search_space - score_under_full_search_space
    elif score_under_constrained_search_space == score_under_full_search_space:
        same_best_score.append(script_object)
        if score_under_constrained_search_space == 100:
            full_percentage_count += 1
        else:
            same_performance_count += 1


print(f"{improved_performance_count} scripts achieve higher accuracy score under constrained search space. Average increase is {sum(improved.values())/improved_performance_count: .2f}%")
print(f"{decreased_performance_count} scripts actually suffer decreased accuracy score. Average decrease is Average increase is {sum(decreased.values())/decreased_performance_count: .2f}%")
print(f"{same_performance_count + full_percentage_count} scripts have same accuracy score, out of which {full_percentage_count} reaches 100 percent accuracies" )


# Compute how many scripts reached best hyperparameter configuration earlier.

earlier = {}
later = {}
same = {}
for i, script_object in enumerate(same_best_score):

    score_under_full_search_space = script_object["best_score"]
    accuracies_under_full_search_space = script_object["accuracies"]
    full_search_space_idx = float('inf')
    for idx, accuracy_full_search_space in enumerate(accuracies_under_full_search_space):
        if score_under_full_search_space == accuracy_full_search_space:
            full_search_space_idx = idx

    score_under_constrained_search_space = script_object["constrained_search_space"]["best_score"]
    accuracies_under_constrained_search_space = script_object["constrained_search_space"]["accuracies"]
    constrained_search_space_idx = float('inf')
    for idx, accuracy_constrained_search_space in enumerate(accuracies_under_constrained_search_space):
        if score_under_constrained_search_space == accuracy_constrained_search_space:
            constrained_search_space_idx = idx

    if full_search_space_idx == float('inf') or constrained_search_space_idx == float('inf'):
        raise ValueError("Check") 

    if full_search_space_idx > constrained_search_space_idx:
        earlier[i] = full_search_space_idx - constrained_search_space_idx
    elif full_search_space_idx < constrained_search_space_idx:
        later[i] = full_search_space_idx - constrained_search_space_idx
    else:
        same[i] = 0
    

print(f"Out of {len(same_best_score)} that has the same best score.")
print(f"{len(earlier)} scripts achieve best score earlier under constrained search space. Average increase is {sum(earlier.values())/len(earlier): .2f}")
print(f"{len(later)} scripts achieve best score slower under constrained search space. Average decrease {sum(later.values())/len(later): .2f}")
print(f"{len(same)} scripts have reached best score at the same iteration" )




