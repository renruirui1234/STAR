import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from stgcn import STGCN
from stgcn_utils import *

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


use_gpu = False
num_timesteps_input = 12
num_timesteps_output = 3

epochs = 1000
batch_size = 25

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()

args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        # indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[i:i + batch_size], training_target[i:i + batch_size]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        X_batch_cpu=X_batch.detach().cpu().numpy()
        out,A = net(A_wave, X_batch)
        try:
            outs=np.concatenate((outs,out.detach().cpu().numpy()),axis = 0)
        except:
            outs=np.array(out.detach().cpu().numpy())
        y_batch_cpu=y_batch.detach().cpu().numpy()
        out_cpu=torch.max(out,dim=1)[1]
        out_cpu=out_cpu.detach().cpu().numpy()
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    print("Training epoch_loss: {}".format(sum(epoch_training_losses)/len(epoch_training_losses)))
    return sum(epoch_training_losses)/len(epoch_training_losses),outs



def val_epoch(val_input, val_target, batch_size):
    """
    Trains one epoch with the given data.
    :param val_input: val inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param val_target: val targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during val.
    :return: Average loss for this epoch.
    """





#如何考虑时序和空域的正则化
if __name__ == '__main__':
    min_loss=-1
    save_root='./checkpoints/'
    torch.manual_seed(7)
    state={}

    explain_class = 'cpu'

    X_dir="./anomaly_instance/total_node_"+explain_class+".npy"
    label_dir="./anomaly_instance/total_label_"+explain_class+".npy"
    A, X,labels, means, stds = load_traffic_data(X_dir,label_dir)

    np.save('./'+explain_class+'-means.npy',means)
    np.save('./'+explain_class+'-stds.npy', stds)

    state = np.random.get_state()

    idx=np.arange(0,len(labels),1)
    np.random.shuffle(idx)

    train_index=int(0.7*len(idx))
    split_line1 = idx[:train_index]
    split_line2=idx[train_index:]

    train_original_data = X[split_line1,:, :, :]
    val_original_data = X[split_line2,:, :, :]

    train_original_target = labels[split_line1]
    val_original_target = labels[split_line2]

    train_original_A = A[split_line1]
    val_original_A = A[split_line2]


    training_input, training_target,training_A = generate_dataset_new(train_original_data,train_original_target,train_original_A)
    val_input, val_target,val_A = generate_dataset_new(val_original_data,val_original_target,val_original_A)


    A=np.sum(A,axis=0)
    # A = A + A.T
    A=np.where(A>0,1,0)
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    A_wave=A_wave.float()
    A_wave = A_wave.to(device=args.device)

    cg={}
    cg['feat']=X[idx]
    cg['label']=labels[idx]
    cg['adj']=A

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                training_input.shape[2],
                A_wave.shape[0]+1).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.5e-3)
    loss_criterion = nn.CrossEntropyLoss()


    training_losses = []
    validation_losses = []
    validation_maes = []

    labels_new=[]

    for epoch in range(epochs):
        loss,outs_train = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)


        # Run validation
        with torch.no_grad():
            net.eval()
            print('test')
            epoch_val_losses = []
            outs = []
            for i in range(0, val_input.shape[0], batch_size):

                X_batch, y_batch = val_input[i:i + batch_size], val_target[i:i + batch_size]
                X_batch = X_batch.to(device=args.device)
                y_batch = y_batch.to(device=args.device)
                X_batch_cpu = X_batch.detach().cpu().numpy()
                out,A = net(A_wave, X_batch)
                outs_train = np.concatenate((outs_train, out.detach().cpu().numpy()), axis=0)
                y_batch_cpu = y_batch.detach().cpu().numpy()
                labels_new.append(y_batch_cpu)
                out_cpu = torch.max(out, dim=1)[1]
                out_cpu = out_cpu.detach().cpu().numpy()
                outs.append(out_cpu)
                out_cpu = torch.max(out, dim=1)[1]
                loss = loss_criterion(out, y_batch)
                epoch_val_losses.append(loss.detach().cpu().numpy())
                # print("val batch_loss: {}".format(epoch_val_losses[-1]))

            test_loss=sum(epoch_val_losses) / len(epoch_val_losses)
            print("val batch_loss: {}".format(sum(epoch_val_losses) / len(epoch_val_losses)))

            for i in outs:
                try:
                    y_pred=np.append(y_pred,i)
                except:
                    y_pred=i
            y_true=val_target.detach().cpu().numpy()


            y_total_true=labels[idx]
            y_total_pred=np.argmax(outs_train,axis=1)

            if epoch%1==0:
                with open("./result.txt", "a") as f:
                    print(len(y_true))
                    print(len(y_pred))
                    f.write(str(f1_score(y_true, y_pred, average='weighted'))+'   ')
                    f.write(str(precision_score(y_true, y_pred, average='weighted'))+'   ')
                    f.write(str(recall_score(y_true, y_pred, average='weighted'))+'\n')
                    print(f1_score(y_true, y_pred, average='weighted'))
                    print(precision_score(y_true, y_pred, average='weighted'))
                    print(recall_score(y_true, y_pred, average='weighted'))

                    print(f1_score(y_total_true, y_total_pred, average='weighted'))
                    print(precision_score(y_total_true, y_total_pred, average='weighted'))
                    print(recall_score(y_total_true, y_total_pred, average='weighted'))

                    f.flush()
                    f.close()


            if epoch % 5 == 0:  # 需要设置一个test
                if min_loss==-1:
                    min_loss=test_loss
                    print("save model")
                    # state={'optimizer':,'model_state','optimizer_state','cg':}
                    state={}
                    state['split_line1']=split_line1
                    # state['split_line2'] = split_line2
                    state['model_state']=net.state_dict()
                    cg['pred']=outs_train
                    state['cg']=cg
                    state['optimizer']=optimizer
                    torch.save(state, save_root + 'STGCN_model_'+explain_class+'.pth')
                    # torch.save(state, save_root + 'STGCN_model_cpu' + '.pth')
                if test_loss < min_loss:
                    min_loss = test_loss
                    print("save model")
                    state['cg']['pred'] = outs_train

                    y_pred=np.argmax(outs_train,axis=1)
                    precision_score(labels, y_pred, average='weighted')

                    state['model_state'] = net.state_dict()
                    state['optimizer'] = optimizer
                    torch.save(state, save_root + 'STGCN_model_'+explain_class+'.pth')
                    # torch.save(state, save_root + 'STGCN_model_cpu' + '.pth')

            del y_pred

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

