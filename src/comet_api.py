import sys
import os

import argparse
from pprint import pprint
import comet_ml
from comet_ml import API
import numpy as np
from pprint import pprint


class CometAPI(object):
    def __init__(self, **kwargs):
        self.project_name = kwargs.get('project-name', '<projectname>')
        self.user_name = kwargs.get('user-name', '<username>')
        self.api = API()

    def get_params(self, expKey):
        exp = self.api.get_experiment(self.user_name, self.project_name, expKey)
        params = exp.get_parameters_summary()
        params = self.get_current_value(params)
        return params

    def get_current_value(self, items):
        key_val = {}
        for item in items:
            key = item['name']
            try:
                value = eval(item['valueCurrent'])
            except:
                value = item['valueCurrent']
                if value.lower() in ['true']:
                    value = True
                elif value.lower() in ['false']:
                    value = False
                elif value.lower() in ['null', 'none']:
                    value = None
            key_val[key] = value
        return key_val

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, help="dataset name", default="ReVerb20K")
    parser.add_argument("--model_name", type=str, help="model name", default='OKGIT')
    parser.add_argument("--project-name",   type=str,   default="<projectname>",   help="project name in comet.ml")
    parser.add_argument("--user-name",   type=str,   default="<username>",   help="user name")
    parser.add_argument("--metric",   type=str,   default="validate_mrr",   help="metric to select best model")
    return parser

def get_key_val(values):
    keyval = {}
    for metric in values:
        keyval[metric['name']] = metric
    return keyval

def get_step_metrics(exp, step):
    step_metrics= {}
    for metric in exp.metrics_raw:
        if metric['step'] == step:
            step_metrics[metric['metricName']] = float(metric['metricValue'])
    return step_metrics

def get_current_value(values):
    new_values = {}
    for key, val in values.items():
        new_values[key] = val['valueCurrent']
    return new_values

def is_relevant_exp(params, conditions):
    #if parameters['dataset']['valueCurrent'] != dataset or parameters['model_name']['valueCurrent'] != model:
    is_relevant = True
    for key, val in conditions.items():
        if key == 'init':
            if val == 'adj':
                if 'adj_all' not in params:
                    # part of previous experiments where adj meant adj_all
                    if params[key]['valueCurrent'] !=  val:
                        is_relevant = False
                else:
                    # adj_all = true means adj
                    # adj_all = false means adj_train
                    if params['adj_all']['valueCurrent'] == 'true' and params[key]['valueCurrent']:
                        pass
            if val == 'adj' and (params[key]['valueCurrent'] != 'adj' or params.get('adj_all', {}).get('valueCurrent', 'true') != 'true'):
                is_relevant = False
            elif val == 'adj_train' and (params[key]['valueCurrent'] != 'adj' or params.get('adj_all',  {}).get('valueCurrent', 'true') != 'false'):
                is_relevant = False
            elif params[key]['valueCurrent'] != val:
                is_relevant = False
        elif params[key]['valueCurrent'] != val:
            is_relevant = False
            break
    return is_relevant

def best_run(conditions, projectname='<projectname>', username='<username>', key_metric='valid_precision_5'):
    api = API()
    exps = api.get(username, projectname)
    rel_exps = []
    rel_values = []
    rel_metrics = []
    rel_params = []
    for exp in exps:
        parameters = get_key_val(exp.parameters)
        if not is_relevant_exp(parameters, conditions):
            continue
        metrics = get_key_val(exp.metrics)
        if key_metric in metrics and key_metric not in ['train_loss']:
            step = metrics[key_metric]['stepMax']
            key_value = float(metrics[key_metric]['valueMax'])
        else:
            # incorporating the difference in step counts
            step = metrics['train_loss']['stepMin'] - 1
            # negate the value so as to choose max later
            key_value = -float(metrics['train_loss']['valueMin'])
        rel_values.append(key_value)
        rel_metrics.append(get_step_metrics(exp, step))
        rel_exps.append(exp)
        rel_params.append(get_current_value(parameters))
    if len(rel_params) == 0:
        raise Exception
    best_idx = np.array(rel_values).argmax()
    best_exp = rel_exps[best_idx]
    best_params = rel_params[best_idx]
    best_metrics = rel_metrics[best_idx]
    return {'exp': best_exp, 'params': best_params, 'metrics': best_metrics, 'name': best_params['name']}

def main():
    comet = CometAPI()
    params = comet.get_params("<expID>")
    pprint(params)

if __name__ == "__main__":
    main()

