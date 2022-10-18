# coding: utf-8
import os
import numpy as np
import paddle
from paddle.utils.download import get_weights_path_from_url


# def load_dygraph_pretrain(model, path=None):
#     if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
#         raise ValueError("Model pretrain path {}.pdparams does not "
#                          "exists.".format(path))
#     param_state_dict = paddle.load(path + ".pdparams")
#     if isinstance(model, list):
#         for m in model:
#             if hasattr(m, 'set_dict'):
#                 m.set_dict(param_state_dict)
#     else:
#         model.set_dict(param_state_dict)
#     return
#
#
# def load_dygraph_pretrain_from_url(model, pretrained_url, use_ssld=False):
#     if use_ssld:
#         pretrained_url = pretrained_url.replace("_pretrained",
#                                                 "_ssld_pretrained")
#     local_weight_path = get_weights_path_from_url(pretrained_url).replace(
#         ".pdparams", "")
#     load_dygraph_pretrain(model, path=local_weight_path)
#     return
#
#
# def _load_pretrained(pretrained, model, model_url, use_ssld=False):
#     if pretrained is False:
#         pass
#     elif pretrained is True:
#         load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
#     elif isinstance(pretrained, str):
#         load_dygraph_pretrain(model, pretrained)
#     else:
#         raise RuntimeError(
#             "pretrained type is not available. Please use `string` or `boolean` type."
#         )


def get_param_pd(model):
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


def _load_pretrained(pretrained, model, model_url):
    import json
    # print(paddle.summary(model, [(1, 3, 224, 224)]))
    if pretrained is False:
        pass
    elif pretrained is True:
        # weight_path = get_path_from_url(model_url, '../model')
        weight_path = get_weights_path_from_url(model_url)
        # print(weight_path)
        param = paddle.load(weight_path)
        get_param_pd(model)

        # for val in param.items():  # paddle
        #     print(val[0], val[1].shape)
        # model_state = list(model.state_dict().keys())  # paddle
        # print(model_state)
        # pd2tlx_weight = {}
        # for i in range(len(model_state)):
        #     model_key = model_state[i]
        #     pd2tlx_weight.update({model_key: i})
        # with open('pd2tlx_weight.json', 'w') as f:
        #     f.write(json.dumps(pd2tlx_weight))

        model.set_dict(param)
