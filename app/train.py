import boto3
import os
session = boto3.Session(
aws_access_key_id = "",
aws_secret_access_key = "",
)
s3 = session.resource('s3')
s3client = session.client('s3')
bucket = 'coco-captions'
keys = ['train_immap.json','val_immap.json',
        'annotations/captions_train2014.json',
        'annotations/captions_val2014.json','train.npy','val.npy',
        'vocab_100k.pt']
for key in keys:
    path = f"data/{key}"
    if not os.path.exists(path):
        s3client.download_file(bucket,key,path)
        print(key)

import torch
import torch.nn as nn
import torchvision
import efficientnet_pytorch
import transformers
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import json

base_path = "data"

with open(f'{base_path}/annotations/captions_train2014.json','r') as f:
    annot_train = json.load(f)

with open(f'{base_path}/annotations/captions_val2014.json','r') as f:
    annot_val = json.load(f)

LONGEST_CAPTION = max(len(d['caption'].split())
                      for d in annot_train['annotations']+annot_val['annotations'])
LONGEST_CAPTION

train_pics = np.load(f'{base_path}/train.npy')
val_pics = np.load(f'{base_path}/val.npy')

with open(f'{base_path}/train_immap.json','r') as f:
    train_immap = json.load(f)

train_fn_to_index = {key.split('/')[1]:val for key,val in train_immap.items()}
train_index_to_fn = {val:key for key,val in train_fn_to_index.items()}

with open(f'{base_path}/val_immap.json','r') as f:
    val_immap = json.load(f)

val_fn_to_index = {key.split('/')[1]:val for key,val in val_immap.items()}
val_index_to_fn = {val:key for key,val in val_fn_to_index.items()}

train_imfn_to_imid = {d['file_name']:d['id'] for d in annot_train['images']}
train_imid_to_caption = {d['image_id']:d['caption'] for d in annot_train['annotations']}
train_imfn_to_caption = {fn:train_imid_to_caption[id_] for fn,id_ in train_imfn_to_imid.items()}

val_imfn_to_imid = {d['file_name']:d['id'] for d in annot_val['images']}
val_imid_to_caption = {d['image_id']:d['caption'] for d in annot_val['annotations']}
val_imfn_to_caption = {fn:val_imid_to_caption[id_] for fn,id_ in val_imfn_to_imid.items()}

vocab = torch.load(f"{base_path}/vocab_100k.pt")

UNK_TOK = '~~UNK~~'
vocab.itos = np.concatenate([[UNK_TOK],vocab.itos])
vocab.stoi = {v:k+1 for v,k in vocab.stoi.items()}
vocab.stoi[UNK_TOK] = 0 ## ends up being 1
vocab.vectors = torch.cat([torch.zeros(1,300),vocab.vectors])

END_TOK = '~~END~~'
vocab.itos = np.concatenate([[END_TOK],vocab.itos])
vocab.stoi = {v:k+1 for v,k in vocab.stoi.items()}
vocab.stoi[END_TOK] = 0
vocab.vectors = torch.cat([torch.zeros(1,300),vocab.vectors])

train_captions = torch.zeros(train_pics.shape[0],LONGEST_CAPTION+1,dtype=torch.long)
train_loss_mask = torch.zeros(train_pics.shape[0],LONGEST_CAPTION+1,dtype=torch.bool)
for i in range(train_pics.shape[0]):
    caption = train_imfn_to_caption[train_index_to_fn[i]]
    split = caption.split()
    for word_ind, word in enumerate(split):
        word = word.lower().replace('.','').replace(',','').replace(';','')
        if not word:
            continue
        if word in vocab.stoi:
            train_captions[i, word_ind] = vocab.stoi[word]
        else:
            #print(word)
            train_captions[i, word_ind] = vocab.stoi[UNK_TOK]
        train_loss_mask[i, word_ind] = True
    train_captions[i, word_ind + 1] = vocab.stoi[END_TOK] ## cause it should generate end tok
    train_loss_mask[i, word_ind + 1] = True

val_captions = torch.zeros(val_pics.shape[0],LONGEST_CAPTION+1,dtype=torch.long)
val_loss_mask = torch.zeros(val_pics.shape[0],LONGEST_CAPTION+1,dtype=torch.bool)
for i in range(val_pics.shape[0]):
    caption = val_imfn_to_caption[val_index_to_fn[i]]
    split = caption.split()
    for word_ind, word in enumerate(split):
        word = word.lower().replace('.','').replace(',','').replace(';','')
        if not word:
            continue
        if word in vocab.stoi:
            val_captions[i, word_ind] = vocab.stoi[word]
        else:
            val_captions[i, word_ind] = vocab.stoi[UNK_TOK]
        val_loss_mask[i, word_ind] = True
    val_captions[i, word_ind + 1] = vocab.stoi[END_TOK]
    val_loss_mask[i, word_ind + 1] = True

batch_size = 200

class Dataset(torch.utils.data.Dataset):
    def __init__(self, pics, captions, loss_mask, pic_transform):
        self.pics = pics
        self.captions = captions
        self.loss_mask = loss_mask
        self.pic_transform = pic_transform
    
    def __len__(self):
        return self.pics.shape[0]
    
    def __getitem__(self, idx):
        return {'pics':self.pic_transform(self.pics[idx]),
                'captions':self.captions[idx],
                'loss_mask':self.loss_mask[idx]}

class AddGaussianNoise():
    def __init__(self, mean=0., std=.25):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
train_transform = torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  normalize, AddGaussianNoise()
                                                 ])
train_extra_inds = np.random.permutation(len(val_pics))[:40000]
train_extra_inds_s = set(train_extra_inds)
val_inds = np.array([i for i in range(len(val_pics)) if i not in train_extra_inds_s])
train_ds = Dataset(np.vstack([train_pics,val_pics[train_extra_inds]]),
                   np.vstack([train_captions,train_captions[train_extra_inds]]),
                   np.vstack([train_loss_mask,train_loss_mask[train_extra_inds]]), train_transform)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                normalize])
val_ds = Dataset(val_pics[val_inds], val_captions[val_inds], val_loss_mask[val_inds], val_transform)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
class Captioner(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.word_emb_size = 300
        self.encoder = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        self.pic_emb_size = 1280
        self.average_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=.2)
        self.decoder = nn.GRU(input_size=self.word_emb_size,
                              hidden_size=self.pic_emb_size,
                              batch_first=True)
        self.classifier = nn.Linear(self.pic_emb_size,self.vocab_size)
        self.start_tok_embed = nn.Parameter(torch.randn(self.word_emb_size,
                                                        dtype=torch.float32),requires_grad=True)
        
    def forward(self, ims, caption_embs):
        bs = ims.size(0)

        im_embs = self.encoder.extract_features(ims)
        im_embs = self.average_pooling(im_embs).view(bs,self.pic_emb_size)
        hidden = im_embs.unsqueeze(0)

        caption_embs = torch.cat([self.start_tok_embed.expand(bs,1,self.word_emb_size),caption_embs],axis=1)
        out, _ = self.decoder(caption_embs,hidden)
        out = self.dropout(out.reshape(bs*caption_embs.size(1),-1))
        out = self.classifier(out)
        out = out.view(bs,caption_embs.size(1),-1)
        return out 

def get_word_embs(vocab, word_inds):
    words = word_inds[:,:-1]
    size = words.size()
    return vocab.vectors[words.flatten()].view(size[0],size[1],300)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Captioner(vocab)
model = model.to(device)
for param in model.encoder.parameters():
    param.requires_grad = False
model.encoder.eval()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
            factor=0.1, patience=0, verbose=True, 
            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

for epoch in range(999):

    train_losses = []
    val_losses = []

    model.train()
    model.encoder.eval()

    for i,batch in enumerate(train_dl):
        pics = batch['pics'].to(device)
        caption_embs = get_word_embs(vocab, batch['captions']).to(device)
        loss_mask = batch['loss_mask'].to(device)
        preds = model(pics, caption_embs)[loss_mask]
        labels = batch['captions'].to(device)[loss_mask]
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())
    
    model.eval()

    with torch.no_grad():
        for i,batch in enumerate(val_dl):
            pics = batch['pics'].to(device)
            caption_embs = get_word_embs(vocab, batch['captions']).to(device)
            loss_mask = batch['loss_mask'].to(device)
            preds = model(pics, caption_embs)[loss_mask]
            labels = batch['captions'].to(device)[loss_mask]
            loss = criterion(preds,labels)
            val_losses.append(loss.item())

    print(f"epoch: {epoch}, tr_loss: {np.mean(train_losses)}, "
          f"vl_loss: {np.mean(val_losses)}")

    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break

    if lr<1e-5:
        break

for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-3

model.train()
model.encoder.eval()

for i,batch in enumerate(val_dl):
    pics = batch['pics'].to(device)
    caption_embs = get_word_embs(vocab, batch['captions']).to(device)
    loss_mask = batch['loss_mask'].to(device)
    preds = model(pics, caption_embs)[loss_mask]
    labels = batch['captions'].to(device)[loss_mask]
    loss = criterion(preds,labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save(model.state_dict(), 'weights.pt')

