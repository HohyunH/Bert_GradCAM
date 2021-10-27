import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
from transformers import BertTokenizer, BertModel

import re
import os
import time
import argparse
import math
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

from dataloader import *
from model import *
from gradcam import GradCam


def check_kld(heatmap):
    neg = sorted(list(np.where(heatmap > 0, 0, heatmap)[0]))
    pos = sorted(list(np.where(heatmap < 0, 0, heatmap)[0]), reverse=True)
    pos_num = [p for p in pos if p != 0]
    neg_num = [abs(n) for n in neg if n != 0]

    pad_length = [len(pos_num), len(neg_num)][np.argmin([len(pos_num), len(neg_num)])]

    pos_num = pos_num[:pad_length];
    neg_num = neg_num[:pad_length]

    def kl_divergence(p, q):
        return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)))

    return kl_divergence(pos_num, neg_num)


def extract_with_cam(grad_cam, inputs, masks, device, word_dict, vocab):

    sentence = inputs.to(device)
    masks = masks.to(device)

    sentence = sentence.unsqueeze(0)
    masks = masks.unsqueeze(0)

    pred, output = grad_cam(sentence, masks).max(dim=-1)  # 마지막 class node중, max가져옴.
    grad_cam.zero_grad()  # 역전파값 모두 초기화
    pred.backward(retain_graph=True)  # 역전파 계산. print찍어보면 이미 cahce 다 비웠음.

    # register_hook으로 중간 gradient 붙잡아둠.
    gradients = grad_cam.get_activations_gradient()
    '''
        detach() 현재 계산 기록으로 분리됨. 이후에
        detach()는 in-place함수가 아니라 requires_grad, grad_fn이 각각 False, None인 "새로운" Tensor를 리턴한다!
    '''
    activations = grad_cam.get_activations(sentence, masks).detach()

    # global average pooling : 각 채널별로 평균 구함.
    pooled_gradients = torch.mean(gradients, dim=[0, 2])
    for k in range(gradients.shape[1]):
        activations[:, k, :] *= pooled_gradients[k]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = heatmap.view(1, -1).cpu().numpy()
    heatmap = cv2.resize(heatmap, dsize=(512, 1))
    heatmap = np.multiply(heatmap, masks.cpu().numpy())
    standard_score = check_kld(heatmap)

    # gradients.remove()
    # pooled_gradients.remove()
    sentences_list = []
    att_score_list = []
    tmp_sentence = ""
    att_score = 0
    for i, word in enumerate(sentence[0]):
        if word != word_dict['[SEP]'] and i != len(
                list(sentence[0])) - 1:  ## SEP가 없는 단어 추가 : 이유가 뭘까= > 문장 토큰화가 안되는 문장임이 분명하다,,,,,
            if word == 0:
                break
            if vocab[word][0] != '#':
                tmp_sentence += vocab[word] + ' '
                att_score += heatmap[0][i]
            else:
                tmp_sentence += '' + vocab[word][2:]
                att_score += heatmap[0][i]
        else:
            sentences_list.append(tmp_sentence)
            tmp_sentence = ""
            att_score_list.append(att_score)
            att_score = 0

    if standard_score > 2:
        max_idx = np.argmax(att_score_list)
        min_idx = np.argmin(att_score_list)

        return sentences_list[max_idx], sentences_list[min_idx]

    else:
        att_score_list = [abs(a) for a in att_score_list]
        max_idx = np.argmax(att_score_list)

        return sentences_list[max_idx]

def validation(network, val_loader, device):
  # Test the model
  network.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

  with torch.no_grad():
      correct = 0
      total = 0
      for i, (inputs, labels, masks, _) in enumerate(val_loader):
          tests = inputs.to(device)
          test_labels = labels.long().to(device)
          masks = masks.to(device)

          outputs = network(tests, masks)
          _ , predicted = torch.max(outputs.data , 1)

          total += test_labels.size(0)
          correct += (predicted == test_labels).sum().item()

      print('Test Accuracy of the model {} %'.format(100* correct/total))

      return correct/total


def alpha_weight(step, T1=100, T2=700, af=3):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
        return ((step - T1) / (T2 - T1)) * af


def masked_loss(pred, target, mask, criterion):
    loss = 0.0
    num = 0
    for i in range(len(pred)):
        if True in mask[i]:
            loss += criterion(pred[i].unsqueeze(0), target[i].view(-1))
            num += 1
    return loss / num


def semisup_train(network, grad_cam, train_loader, test_loader, val_loader, criterion, optimizer, device, word_dict, vocab, tokenizer,
                  EPOCHS=1, label_batch=250, max_len=512, masked=True, xai_condition=True):
    with torch.autograd.set_detect_anomaly(True):

        # Instead of using current epoch we use a "step" variable to calculate alpha_weight
        # This helps the model converge faster
        acc_test = []
        step = 100
        best_epoch_acc = 0
        network.train()
        for epoch in range(EPOCHS):
            print(f"{epoch + 1} epochs running...")
            check_pseudo = []
            for batch_idx, (inputs, labels, masks, _) in enumerate(tqdm(test_loader)):

                # Forward Pass to get the pseudo labels
                x_unlabeled = inputs.to(device)
                un_masks = masks.to(device)
                network.eval()
                output_unlabeled = network(x_unlabeled, un_masks)
                _, pseudo_labeled = torch.max(output_unlabeled, 1)
                network.train()

                # Now calculate the unlabeled loss using the pseudo label
                output = network(x_unlabeled, un_masks)
                pseudo_labeled = pseudo_labeled.type(torch.LongTensor).to(device)
                # pseudo_labeled = pseudo_labeled.unsqueeze(1)

                output = output.type(torch.FloatTensor).to(device)
                soft_max = F.softmax(output, dim=1)
                soft_max = torch.max(soft_max, dim=1)[0]
                # Data Filtering
                if masked == True:
                    total_masked = soft_max.ge(0.8)  ## upper condition : 0.8

                    if True not in total_masked:
                        pass
                    else:
                        unlabeled_loss = alpha_weight(step) * masked_loss(output, pseudo_labeled, total_masked, criterion)
                        optimizer.zero_grad()
                        unlabeled_loss.backward()
                        optimizer.step()

                        ########################################################################
                        if xai_condition:

                            sentences = []
                            for i, (sentence, mask) in enumerate(zip(x_unlabeled, un_masks)):
                                if len(sentence * total_masked[i]) > 0:
                                    if list(sentence * total_masked[i]).count(0) == len(
                                            list(sentence * total_masked[i])):  ## 곱했을 때 전부 0이 되는 경우 제외
                                        pass
                                    else:
                                        sentences.append([sentence * total_masked[i], mask * total_masked[i]])

                            pseudo_data = []
                            for inputs, masks in sentences:
                                cam_datas = extract_with_cam(grad_cam, inputs, masks, device, word_dict, vocab)
                                if len(cam_datas) == 2:
                                    pseudo_data.append(cam_datas[0]);
                                    pseudo_data.append(cam_datas[1])
                                else:
                                    pseudo_data.append(cam_datas)

                            to_token = [get_token(x, tokenizer) for x in pseudo_data]
                            xai_input = torch.tensor([get_ids(x, max_len, tokenizer) for x in to_token])
                            xai_masks = torch.LongTensor(get_mask(xai_input))

                            xai_input = xai_input.to(device)
                            xai_masks = xai_masks.to(device)

                            check_pseudo.append(xai_input.size(0))  ## pseudo label 갯수 체크

                            if xai_input.size(0) == 0:  ## 뽑아낸게 없으면 넘어가고
                                pass

                            else:
                                network.eval()
                                pseudo_output = network(xai_input, xai_masks)
                                _, pseudo_label = torch.max(pseudo_output, 1)

                                network.train()
                                output_xai = network(xai_input, xai_masks)
                                pseudo_label = pseudo_label.type(torch.LongTensor).to(device)
                                output_xai = output_xai.type(torch.FloatTensor).to(device)

                                xai_soft_max = F.softmax(output_xai, dim=1)

                                xai_total_masked = xai_soft_max.ge(0.8)

                                if True not in xai_total_masked:
                                    pass

                                else:
                                    xai_loss = alpha_weight(step - 100) * masked_loss(output_xai, pseudo_label,
                                                                                      xai_total_masked, criterion)
                                    optimizer.zero_grad()
                                    xai_loss.backward()
                                    optimizer.step()

                ########################################################################
                else:
                    unlabeled_loss = alpha_weight(step) * criterion(output, pseudo_labeled)

                    optimizer.zero_grad()
                    unlabeled_loss.backward()
                    optimizer.step()

                # For every "label_batch" batches train one epoch on labeled data
                if batch_idx % label_batch == 0:
                    print("Training Labeled Data.....")
                    # Normal training procedure
                    for i, (inputs, labels, masks, _) in enumerate(train_loader):
                        inputs = inputs.to(device)
                        labels = labels.long().to(device)
                        masks = masks.to(device)
                        outputs = network(inputs, masks)
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    acc = validation(network, val_loader, device)
                    print(f"xai pusedo data : {len(check_pseudo)}")
                    acc_test.append(acc)
                    if acc > best_epoch_acc:
                        best_epoch_acc = acc
                        print(f"******* BEST ACCURACY : {best_epoch_acc} in [{batch_idx} batch] *******")
                    # Now we increment step by 1
                    step += 1
                    check_pseudo = []

    return acc_test

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_len', type=int, default=512, help='determining the length of the sequence')
    parser.add_argument('--epochs', type=int, default=1, help='determining the epoch size for model training')
    parser.add_argument('--batch', type=int, default=16, help='determining the batch size for model training')
    parser.add_argument('--label_batch', type=int, default=200, help='determining the number of batch size when train labeled data')
    parser.add_argument('--masked', type=bool, default=True, help='determining use mask loss or not')
    parser.add_argument('--xai', type=bool, default=True, help='determining use proposed method or not')
    parser.add_argument('--test_num', type=int, default=10000, help='determining the number of unlabelled data')
    parser.add_argument('--class_num', type=int, default=100, help='determining the number of data for each class')
    parser.add_argument('--gpu_set', type=str, default="0,1,2,3", help='determining the number of gpu and input those indexes')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_set

    torch.manual_seed(777)
    random.seed(777)
    np.random.seed(777)

    yelp = pd.read_csv('./yelp.csv')
    yelp['label'] = yelp['rating'] - 1
    del yelp['rating']

    class_num = args.class_num
    train_df = yelp[:1000]
    test_df = yelp[1000:-1000]
    val_df = yelp[-1000:]

    train_df = class_same(train_df, class_num)
    test_df_1 = test_df[:args.test_num]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = list(tokenizer.vocab.keys())
    word_dict = dict(tokenizer.vocab.items())
    BATCH = args.batch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Available devices ', torch.cuda.device_count())

    train_data = BERT_CNN(train_df, tokenizer)
    train_loader = DataLoader(train_data, batch_size=BATCH, num_workers=2, shuffle=True)

    test_data = BERT_CNN(test_df_1, tokenizer)
    test_loader = DataLoader(test_data, batch_size=BATCH, num_workers=2, shuffle=False)

    val_data = BERT_CNN(val_df, tokenizer)
    val_loader = DataLoader(val_data, batch_size=BATCH, num_workers=2, shuffle=False)

    network = Net(len(train_df.label.unique()))

    # if torch.cuda.device_count() > 1:
    #     network = nn.DataParallel(network).to(device)
    # else:
    network = network.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    grad_cam = GradCam(network, 12).to(device)

    semisup_train(network, grad_cam, train_loader, test_loader, val_loader, criterion, optimizer, device, word_dict, vocab, tokenizer,
                  EPOCHS=args.epochs, label_batch=args.label_batch, max_len=args.max_len, masked=args.masked, xai_condition=args.xai)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()