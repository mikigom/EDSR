import json
import os
import inspect


def parse_flags(FLAGS):
    flags_list = inspect.getmembers(FLAGS)

    flags = dict()
    for field_name, field_element in flags_list:
        flags[field_name] = field_element

    return flags


def save_dict(save_dir, dict):
    try:
        # Tensorflow 1.6
        with open(os.path.join(save_dir, 'model_config.json'), 'w') as fp:
            json.dump(dict, fp)
    except TypeError:
        # Tensorflow 1.6 <
        with open(os.path.join(save_dir, 'model_config.json'), 'w') as fp:
            json.dump(dict['__dict__']['__flags'], fp)

    #"""
    #with open(os.path.join(save_dir, 'model_config.json'), 'w') as fp:
    #    json.dump(dict['__flags'], fp)
    #"""

def load_flags(FLAGS):
    try:
        load_dir = FLAGS.restore_dir
    except AttributeError:
        load_dir = FLAGS.ckpt_dir

    with open(os.path.join(load_dir, 'model_config.json'), 'r') as fp:
        dict = json.load(fp)

    for key, value in dict.items():
        try:
            setattr(FLAGS, key, value)
        except:
            pass
