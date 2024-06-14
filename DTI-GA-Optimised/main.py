import sys
from src.edit import weighted_edit_distance
from datetime import datetime
from src.edit import make_store_matrix
from src.predict import predict
from src.exhaustive_search import exhaustive_search
from src.genetic import Genetic

# calculate edit distance between 2 strings -----
if sys.argv[1]=="1":

    cost = (1, 1, 1) # default for levenstein (icost, dcost, rcost)
    st_time = datetime.now()

    print("\nFINDING THE EDIT DISTANCE:")
    print(f"From {sys.argv[2]} to {sys.argv[3]}:", weighted_edit_distance(sys.argv[2], sys.argv[3], *cost))
    en_time = datetime.now()
    print(f'Time taken: {round((en_time-st_time).total_seconds()*1000)} ms\n')

# calculate weighted edit distance between 2 strings -----
if sys.argv[1]=="2":

    cost = (float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])) # for levenstein (icost, dcost, rcost)
    if not (0<cost[0]<=1 and 0<cost[1]<=1 and 0<cost[2]<=1):
        print("ERROR: all weights must be 0<w<=1")
        exit(1)
    st_time = datetime.now()

    print("\nFINDING THE WEIGHTED EDIT DISTANCE:")
    print(f"From {sys.argv[2]} to {sys.argv[3]} for {cost}:", weighted_edit_distance(sys.argv[2], sys.argv[3], *cost))
    en_time = datetime.now()
    print(f'Time taken: {round((en_time-st_time).total_seconds()*1000)} ms\n')

# calculate auc-roc for given weights  -----
if sys.argv[1]=="3":

    cost = (float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])) # weights for auc-roc
    if not (0<cost[0]<=1 and 0<cost[1]<=1 and 0<cost[2]<=1):
        print("ERROR: all weights must be 0<w<=1")
        exit(1)
    st_time = datetime.now()

    print("\nFINDING THE AUC-ROC:")
    make_store_matrix(*cost)
    auc = predict()
    print(f"AUC-ROC for {cost}:", auc)
    en_time = datetime.now()
    print(f'Time taken: {round((en_time-st_time).total_seconds()*1000)} ms\n')

# running exhaustive search algorithm  -----
if sys.argv[1]=="4":

    n = int(sys.argv[2])
    st_time = datetime.now()

    print("\nRUNNING THE EXHAUSTIVE SEARCH:")
    optimal_weights, max_auc = exhaustive_search(n)
    en_time = datetime.now()
    print(f"The Exhaustive Search for {n} divisions : optimal icost {optimal_weights[0]}, dcost {optimal_weights[1]}, rcost {optimal_weights[2]}, max_auc {round(max_auc, 5)}")
    print(f'Time taken: {round((en_time-st_time).total_seconds())} s\n')

if sys.argv[1]=="5":

    generations = int(sys.argv[2])
    st_time = datetime.now()

    print("\nRUNNING THE GENETIC ALGORITHM:")
    genetic = Genetic(maxGeneration=generations)
    optimal_weights, max_auc = genetic.run()
    en_time = datetime.now()
    print(f"The Genetic Algorithm ran for {generations} generations : optimal icost {optimal_weights[0]}, dcost {optimal_weights[1]}, rcost {optimal_weights[2]}, max_auc {round(max_auc, 5)}")
    print(f'Time taken: {round((en_time-st_time).total_seconds())} s\n')

