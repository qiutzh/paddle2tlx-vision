# coding: utf-8
from tensorlayerx.files import assign_weights


def restore_model(param, model):
    weights = []
    for val in param.items():
        weights.append(val[1])
        if len(model.all_weights) == len(weights):
            break
    # assign weight values
    assign_weights(weights, model)
    del weights
