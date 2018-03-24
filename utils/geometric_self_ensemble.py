import numpy as np


def generate_geometric_self_ensemble_examples(image):
    # image shape should [1, h, w, c] or [h, w, c]
    image_array_list = []

    for k in range(0, 4):
        image_array_list.append(
                np.rot90(image, k=k)
        )

    flipped_image = image[:, ::-1]

    for k in range(0, 4):
        image_array_list.append(
                np.rot90(flipped_image, k=k)
        )

    return image_array_list


def merge_results_geometric_self_ensemble(self_ensemble_results):
    merged_results_list = []
    for i in range(0, 4):
        merged_results_list.append(
                np.rot90(self_ensemble_results[i], k=-i)
        )

    for i in range(0, 4):
        merged_results_list.append(
                np.rot90(
                    self_ensemble_results[4+i],
                    k=-i
                )[:, ::-1]
        )

    merged_results = np.stack(merged_results_list, axis=0)

    return np.mean(merged_results, axis=0)
