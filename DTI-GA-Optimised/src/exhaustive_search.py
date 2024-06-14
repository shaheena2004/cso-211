from src.edit import make_store_matrix
from src.predict import predict
import numpy as np
from src.dataset_names import DATASET_NAME

def exhaustive_search(n):
    weights = np.linspace(0, 1, n + 1)
    optimum_costs = [0, 0, 0]
    max_auc = float("-inf")

    weights_vs_auc = np.genfromtxt(f"./output/weight_vs_auc_{DATASET_NAME}.csv", delimiter=",", dtype=np.float32)

    for icost in  weights:
        for dcost in weights:
            for rcost in weights:
                if 0 in [icost, dcost, rcost]:
                    continue
                make_store_matrix(icost, dcost, rcost)
                auc = predict()
                if not any(np.array_equal(row, np.array((icost, dcost, rcost, auc))) for row in weights_vs_auc):
                    weights_vs_auc = np.vstack((weights_vs_auc, np.array((icost, dcost, rcost, auc))))

                if auc > max_auc:
                    optimum_costs = [icost, dcost, rcost]
                    max_auc = auc


    np.savetxt(f"./output/weight_vs_auc_{DATASET_NAME}.csv", weights_vs_auc , delimiter=",")
    
    return (optimum_costs, max_auc)


