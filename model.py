import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class Net(nn.Module):
    '''
        encoded_layer : [layer, batch, sentence_len, embed_dim] # Bert 12-layer 결과중 하나 선택.
                        Conv1d 의 input로 들어가기 위해, [batch,sentence_len, embed_dim].unsqueeze(1)
                                                     => [batch, 1, sentence_len, embed_dim ]

        Conv1d input shape : 3D or 4D
        (sentence x embedding_dim) : 138 x 768 => (kernel_size = (3,768), padding=1)
                                     138 x 1 로 변환, padding로 인해 138유지.

        1짜리 차원 x.squeeze(3)    : [batch, 16, 138 , 1] =>  [batch, 16 , 138]


        Conv1d input : [batch, 1(number of channel) , sentence_len, emb_dim]
        Conv1d ouput : [batch, out_channel_num , sentence_len, emb_dim]

        [batch, out_channel_num , sentence_len, emb_dim].squeeze(3)
        => [batch, out_channel_num , sentence_len]
    '''

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = nn.Conv1d(1, 16, kernel_size=(3, 768))

        self.cnn = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(16, 16, kernel_size=(3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=(3), padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, kernel_size=(3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=(3), padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=(3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=(3), padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=(3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1 * 31 * 128, 160),  # width * height * channel_num
            nn.Linear(160, num_classes)
        )
        self.relu = nn.LeakyReLU()

    def forward(self, inputs, masks):
        with torch.no_grad():
            encoded_layer = self.bert(inputs, masks)[0]
        encoded_layer = encoded_layer.unsqueeze(1)
        x = self.conv1(encoded_layer)
        x = x.squeeze(3)
        x = self.cnn(x)
        x = x.view(-1, 128 * 31)
        x = self.classifier(x)
        return x