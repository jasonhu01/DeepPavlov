import sys
import pickle
from pathlib import Path
import json
import numpy as np

from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config, build_model_from_config
from deeppavlov.core.commands.utils import expand_path

def get_preds_old(conf):
    # dataset reader and iterator config 
    data = read_data_by_config(conf)
    iterator = get_iterator_from_config(conf, data)
    
    model = build_model_from_config(conf, 'infer', True)
    x, y_true = iterator.get_instances(data_type='valid')
    y_predicted = list(model(list(x)))
    
    model.destroy()
    del x, data, iterator
    
    return y_predicted, y_true


def get_preds(conf, classes, datatype="valid"):
    # dataset reader and iterator config 
    data = read_data_by_config(conf)
    iterator = get_iterator_from_config(conf, data)
    
    model = build_model_from_config(conf, 'infer', True)
    x, y_true = iterator.get_instances(data_type=datatype)
    y_predicted = list(model(list(x)))
    y_pred = [[y_[k] for k in classes] for y_ in y_predicted[1]]
    
    model.destroy()
    del x, data, iterator
    
    return y_pred, y_true


config_path = sys.argv[1]
dict_path = sys.argv[2]

for datatype in ["test", "valid"]:
    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(dict_path, 'r') as f:
        cls_str = f.read()
        
    classes_dict = {k[0]: i for i, k in enumerate([x.split('\t') for x in cls_str.split('\n')][:-1])}
    # classes = list(np.sort(np.array(list(classes_dict.keys()))))
    classes = classes_dict.keys()

    y_pred, y_true = get_preds(config, classes, datatype=datatype)
    result = {}
    result['y_true'] = y_true
    
    result['y_pred'] = y_pred

    with open(expand_path(Path(config["chainer"]["pipe"][0]["save_path"]).parent).joinpath(datatype + '_predictions.pkl'), 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


