import os
import math
import argparse
from itertools import groupby
from operator import itemgetter
import re
import numpy as np
import json
import time
import datetime


parser = argparse.ArgumentParser(description='SUPER')
# parser.add_argument('--train_url', required=False, default=None, help='Location of training outputs.')
# parser.add_argument('--data_url', required=False, default=None, help='Location of data.')
parser.add_argument('--log_url', default='/lustre/grp/gyqlab/linxh/GDPflex/lius/generate-master/BINDTI-main/BINDTI/code_edis/log/7117781.out', help='Location of logs.')
parser.add_argument('--echart_ori_url', default='/lustre/grp/gyqlab/linxh/GDPflex/lius/generate-master/BINDTI-main/BINDTI/code_edis/echarts', help='Location of echarts_loss_logs.')
parser.add_argument('--out_url', default='/lustre/grp/gyqlab/linxh/GDPflex/lius/generate-master/BINDTI-main/BINDTI/code_edis/loss_out', help='output result.')
parser.add_argument('--avg_step', type=int, default=1, help='avg_steps.')
# parser.add_argument('--job_id', type=str, default=None, help='avg_steps.')
parser.add_argument('--refresh_time', type=int, default=None, help='avg_steps.')
parser.add_argument('--show_nodes', type=int, default=1e9, help='show_nodes.')
args_opt = parser.parse_args()
log_url = args_opt.log_url
out_url = args_opt.out_url
avg_step = args_opt.avg_step
# job_id = args_opt.job_id
refresh_time = args_opt.refresh_time
show_nodes = args_opt.show_nodes
echart_ori = args_opt.echart_ori_url

refresh_name = []
while True:
    with open(echart_ori, 'r') as f:
        echart_data = f.readlines()
    
    if log_url.endswith('.log') or log_url.endswith('.out') or log_url.endswith('.txt'):
        with open(log_url, 'r') as f:
            log_data = f.readlines()
    else:
        name_list = os.listdir(log_url)
        name_list = [x for x in name_list if '.log' in x]
        # name_list = ['319515.out', '479071.out']
        log_data = []
        for name in name_list:
            with open(os.path.join(log_url, name), 'r') as f:
                log_data.extend(f.readlines())

    step = 0
    max_loss = 0
    log_dict_all = []
    for h, x in enumerate(log_data):
        if 'Test at Best Model of Epoch' in x or 'dataset:' in x:
            print(x)
            continue
            # x = x.split('epoch')[1]
            # step = int(x.split('= ')[0].strip('\n'))
            loss = float(x.split()[1].strip('\n'))
            loss = float(x.split()[-1].strip('\n'))
            if loss>max_loss:
                max_loss = loss
            log_dict = {"step": step,"total_loss": loss,}
            step += 1
            # if math.isnan(float(tmp[1].strip('\n'))) or int(tmp[0])<100000:
            # if math.isnan(float(tmp[1].strip('\n'))):
            # if math.isnan(float(tmp[1].split(' last_loss ')[1].strip('\n'))):
            # if math.isnan(float(tmp[1].split(' last_loss ')[1].strip('\n'))) or int(tmp[0])<1000:
            #     continue
            # log_dict = {"step": int(tmp[0]),"total_loss": float(tmp[1].strip('\n')),}
            # log_dict = {"step": int(tmp[0]),"total_loss": float(tmp[1].split(' last_loss ')[0]),}
            # log_dict = {"step": int(tmp[0]),"total_loss": float(tmp[1].split(' last_loss ')[1]),}
            log_dict_all.append(log_dict)
    print('max max max max max', max_loss, 'max max max max max')
    log_dict_all.sort(key=itemgetter('step'))
    log_dict_all = groupby(log_dict_all, itemgetter('step'))
    # print('x: ',tmp)
    avg_data = []
    for key, group in log_dict_all:
        count = 0
        tmp = {"step": key, "total_loss": 0, }
        for g in group:
            count += 1
            #g.keys: dict_keys(['rankid', 'step', 'total_loss', 'bert_aa_loss_ipm', 'dmat_loss', 'lddt_loss', 'contact_loss', 'bert_aa_loss', 'chain_aa_loss', 'bert3_token_loss', 'chain_token_loss', 'cluster_loss', 'homology_loss', 'fd_loss'])
            for x in list(g.keys())[1:]:
                tmp[x] += float(g[x])
        for x in list(g.keys())[1:]:
            tmp[x] = tmp[x]/count
        avg_data.append(tmp)
    
    step = [x['step'] + 1 for x in avg_data]
    total_loss = [round(x['total_loss'], 4) for x in avg_data]
    # bert_aa_loss_ipm = [round(x['bert_aa_loss_ipm'], 4) for x in avg_data]
    # mutation_bert_aa_loss = [round(x['mutation_bert_aa_loss'], 4) for x in avg_data]
    # dmat_loss = [round(x['dmat_loss'], 4) for x in avg_data]
    # lddt_loss = [round(x['lddt_loss'], 4) for x in avg_data]
    # contact_loss = [round(x['contact_loss'], 4) for x in avg_data]
    
    def moving_average(a, n=avg_step):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        xx = ret[n - 1:] / n
        return [round(x, 8) for x in xx]

    def pos(a, step, show_nodes=show_nodes):
        a = moving_average(a)
        length = len(a)
        new_array = [[step[i], a[i]] for i in range(length)]
        if show_nodes > length:
            show_nodes = length
        interval = int(length / show_nodes)
        index = list(range(0, length, interval))
        res_array = []
        for inx in index:
            res_array.append(new_array[inx])
        res_array.append(new_array[-1])
        return res_array

    echart_data[25] = str(step)
    echart_data[36] = str(moving_average(total_loss))
    # echart_data[34] = str(pos(total_loss, step))
    # echart_data[40] = str(pos(bert_aa_loss_ipm, step))
    # echart_data[46] = str(pos(mutation_bert_aa_loss, step))
    # echart_data[52] = str(pos(dmat_loss, step))
    # echart_data[58] = str(pos(lddt_loss, step))
    # echart_data[64] = str(pos(contact_loss, step))
    
    for i in range(len(echart_data)):
        if not echart_data[i].endswith('\n'):
            echart_data[i] += '\n'
    with open(out_url, 'w') as f:
        for x in echart_data:
            f.writelines(x)
    # mox.file.copy(src_url=log_path + "echarts", dst_url=out_url)
    print("finished process log file: ", str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), " https://echarts.apache.org/examples/zh/editor.html?c=line-stack")
    if refresh_time:
        time.sleep(refresh_time)
    else:
        print("================================================end")
        break
