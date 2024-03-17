import pandas as pd
import numpy as np
import torch, transformers
from torch import nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
from torch import nn
import torch, transformers
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #참고로 'bert-base-uncased' 모델은 버트의 가장 기본적인 모델을 의미
        #uncased는 모든 문장을 소문자로 대체하겠다는 것 
        
        self.classifier  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768,4)
        )
        
        
    def forward(self,inputs):
        
        #print(inputs['input_ids'].shape)  
        bert_outputs = self.bert(**inputs,return_dict =True)
        #print("bert_outputs",bert_outputs)
        pooler_output = bert_outputs.last_hidden_state[:,0]
        # bert_outputs.last_hidden_state[:,0] -> 첫번째니까 [cls] 토큰의
        # embedding만을 뽑아내어 classification task를 위한 텐서로 변환한다? 
        #print("pooler_output",pooler_output.shape)
        
        logits = self.classifier(pooler_output)
        #print("logits",logits.shape)
        
        return logits


class CustomDataset(Dataset):
    def __init__(self, df):
        self.data =df 
        self.data_list, self.label_list = self.load_data()
        
    def __len__(self):
        return len(self.label_list)
    def load_data(self):
        data_list = self.data['generated']
        label_list = self.data['act']
        
        return data_list, label_list 
    def __getitem__(self, index):
        data = self.data_list[index]
        label = torch.tensor(self.label_list[index], dtype = torch.long)
        return data, label
        

def get_accuracy_by_bert(data,device):
    model = Model()
    model = torch.load('/home/wooseok/peft/dailydialogue/dialogpt/Bert_model/Bert_model_training/epoch12_bert_model.pt')
    model.to(device)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    max_len= 500
    
    data = data[['generated','act']]
    data['act'] = data['act']-1
    test_set = CustomDataset(data)
    test_dl = DataLoader(test_set,batch_size=20, shuffle = False)    
    
    test_acc = 0.0
    test_n_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx,batch in enumerate(tqdm(test_dl, ncols=80, desc='valid_step')):
            data, y_true = batch
            data = tokenizer(list(data), return_tensors='pt', padding=True, truncation=True)
            
            data = { k: v.to(device) for k, v in data.items() }
            y_true = y_true.to(device)
            y_pred = model(data)
            test_acc += torch.sum(y_pred.argmax(1) == y_true).item()
            test_n_samples += len(y_true)
    test_acc = (test_acc / test_n_samples) * 100.
    print(y_pred.argmax(1))
    print("accuracy:",test_acc)
    return test_acc  
def get_generated_sentence_data(model,tokenizer,device,data):
    generated_sentences = []
    for i in tqdm(range(len(data))):
        encoded_context=torch.tensor([data['input_tokens'][i]])
        generated_ids = model.generate(encoded_context.to(device), 
                                    
                                      max_new_tokens = 500,  
                                       pad_token_id=tokenizer.eos_token_id)
        
        generated_sentence= tokenizer.decode(generated_ids[:, encoded_context.shape[-1]:][0], skip_special_tokens=True)
        
        generated_sentences.append(generated_sentence)
    data['generated'] =generated_sentences
    return data
