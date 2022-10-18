# coding: utf-8
import json
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.files import assign_weights


# # assign sequential model weights
# def restore_model(param, model):
#     # # for test
#     # f_out = open("output/param_tnt_pd.json", "w", encoding="utf-8")
#     # for k, v in param.items():
#     #     # print(k, v)
#     #     json_data = {"key": k, "vec_shape": v.shape, "vec_val": v.tolist()}
#     #     f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
#     # f_out.close()
#
#     weights = []
#     for val in param.items():
#         weights.append(val[1])
#         if len(model.all_weights) == len(weights):
#             break
#     # assign weight values
#     assign_weights(weights, model)
#     del weights


def get_param_tlx(model):
    total_params = 0
    trainable_params = 0
    nontrainable_params = 0
    i = 0
    # for p in model.parameters():  # for i, (p, q) in zip(enumerate(model.parameters()), model.named_parameters()):
    for param_name, layer_p in model.named_parameters():
        print(f"{i+1}\t\t{param_name}\t\t{layer_p.name}\t\t{layer_p.shape}")  # print model layer name
        mulValue = np.prod(layer_p.shape)
        total_params += mulValue
        if layer_p.stop_gradient:
            nontrainable_params += mulValue
        else:
            trainable_params += mulValue
        i += 1
    print(f'Total params: {total_params}')
    print(f'Trainable params: {trainable_params}')
    print(f'Non-trainable params: {nontrainable_params}')
    return total_params, trainable_params, nontrainable_params


def restore_model(param, model):
    tlx2pd_namelast = {'filters': 'weight',        # conv2d
                       'biases': 'bias',           # linear
                       'weights': 'weight',        # linear
                       'gamma': 'weight',          # bn
                       'beta': 'bias',             # bn
                       'moving_mean': '_mean',     # bn
                       'moving_var': '_variance',  # bn
                       }
    # print([{i: k} for i, k in model.named_parameters()])
    model_state = [i for i, k in model.named_parameters()]
    weights = []
    # get_param_tlx(model)

    for i in range(len(model_state)):
        model_key = model_state[i]
        model_key_s, model_key_e = model_key.rsplit('.', 1)
        if model_key_e in tlx2pd_namelast:
            new_model_state = model_key_s + '.' + tlx2pd_namelast[model_key_e]
            weights.append(param[new_model_state])
        else:
            print(model_key_e)
    assign_weights(weights, model)
    del weights


def _load_pretrained(pretrained, model, model_url):
    import paddle
    from paddle.utils.download import get_weights_path_from_url

    # print(paddle.summary(model,[(1, 3, 224, 224)]))
    if pretrained is False:
        pass
    elif pretrained is True:
        weight_path = get_weights_path_from_url(model_url)
        # weight_path = os.path.join('../../model', model_url.strip().split('/')[-1])
        param = paddle.load(weight_path)
        get_param_tlx(model)
        # model.set_dict(param)
        restore_model(param, model)
