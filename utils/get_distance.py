import torch
import torch.nn as nn
import numpy as np
class Distance():

    @staticmethod
    def get_distance(model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance

    @staticmethod
    def get_distances_from_current_model(current_model, party_models):
        num_updates = len(party_models)
        distances = np.zeros(num_updates)
        for i in range(num_updates):
            distances[i] = Distance.get_distance(current_model, party_models[i])
        return distances