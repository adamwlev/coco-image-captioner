from flask import (Flask, request, jsonify, make_response, abort,
                  render_template, redirect, url_for, send_from_directory)
import logging, sys, traceback
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torchvision
import efficientnet_pytorch
import transformers
import numpy as np
from PIL import Image
import cv2
from io import BytesIO


app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def pad_image(img, height, width):
    h, w = img.shape[:2]
    t = 0
    b = height - h
    l = 0
    r = width - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

def resize_and_pad(img, height, width, resample=cv2.INTER_AREA):
    if len(img.shape)==2:
        img = np.stack([img,img,img],axis=2)
    target_aspect_ratio = height/width
    im_h, im_w, _ = img.shape
    im_aspect_aspect_ratio = im_h/im_w
    if im_aspect_aspect_ratio>target_aspect_ratio:
        target_height = height
        target_width = int(im_w * target_height/im_h)
    else:
        target_width = width
        target_height = int(im_h * target_width/im_w)
    resized = cv2.resize(img, (target_width, target_height), interpolation=resample)
    return pad_image(resized, height, width)

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            normalize])

def get_lm_score():
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('distilgpt2')
    lm = transformers.GPT2LMHeadModel.from_pretrained('distilgpt2')
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad = False
    max_length = 86
    def lm_score(sents):
        ## sents should be a list of strings
        inds = torch.zeros(len(sents),max_length,dtype=torch.long)
        mask = torch.ones(len(sents),max_length,dtype=torch.float)
        for i in range(len(sents)):
            tok = tokenizer.encode_plus(sents[i], add_special_tokens=True, 
                                        return_tensors='pt', max_length=max_length)['input_ids'][0]
            inds[i, :len(tok)] = tok
            mask[i, len(tok):] = 0
        logits = lm(inds)[0]
        inds_flattened = inds.flatten()
        indexer = torch.arange(0,inds_flattened.size(0),dtype=torch.long)
        chosen_words = logits.view(logits.size(0)*logits.size(1),-1)[indexer,inds_flattened]
        chosen_words = chosen_words.view(logits.size(0),logits.size(1))
        lm_scores = nn.functional.logsigmoid(chosen_words * mask).sum(1).numpy()
        lm_scores /= mask.sum(1).numpy()
        return lm_scores
    return lm_score
    
class Captioner(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.word_emb_size = 300
        self.encoder = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b0')
        self.pic_emb_size = 1280
        self.average_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=.2)
        self.decoder = nn.GRU(input_size=self.word_emb_size,
                              hidden_size=self.pic_emb_size,
                              batch_first=True)
        self.classifier = nn.Linear(self.pic_emb_size,self.vocab_size)
        self.start_tok_embed = nn.Parameter(torch.randn(self.word_emb_size,
                                                        dtype=torch.float32),requires_grad=True)
        self.lm_score = get_lm_score()

    def inference(self, im, num_sample=7, max_length=32, topk=2):
        with torch.no_grad():
            sents = []
            for it in range(num_sample):
                bs = 1
                ims = im.unsqueeze(0)
                im_embs = self.encoder.extract_features(ims)
                im_embs = self.average_pooling(im_embs).view(bs,self.pic_emb_size)
                hidden = im_embs.unsqueeze(0)
                word_emb = self.start_tok_embed.expand(bs,1,self.word_emb_size)
                preds = []
                for i in range(max_length):
                    _, hidden = self.decoder(word_emb, hidden)
                    pred = self.classifier(hidden.squeeze(0)).squeeze()
                    pred = nn.functional.softmax(pred,dim=0)
                    top_preds = torch.topk(pred,topk)
                    top_preds_inds = top_preds.indices.cpu().numpy()
                    top_preds_values = top_preds.values.cpu().numpy()
                    top_preds_values = top_preds_values[top_preds_inds!=1]
                    top_preds_inds = top_preds_inds[top_preds_inds!=1]
                    top_preds_values = top_preds_values/top_preds_values.sum()
                    pred = np.random.choice(top_preds_inds,p=top_preds_values)
                    if pred==0:
                        break
                    word_emb = self.vocab.vectors[pred].view(bs,1,self.word_emb_size)
                    preds.append(self.vocab.itos[pred])
                sents.append(' '.join(preds))
            scores = self.lm_score(sents)
            logger.info(sents)
            return sents[np.argmax(scores)]

vocab = torch.load('/project/app/weights/vocab_100k.pt')

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

model = Captioner(vocab)
model.load_state_dict(torch.load('/project/app/weights/weights.pt',map_location='cpu'))
model.eval()
for param in model.parameters():
    param.requires_grad = False


@app.route("/cocopredict",methods=["POST"])
def predict():

    file = request.files['pic']
    if not file:
        return jsonify("no pic found")
    
    try:
        res = BytesIO(file.read())
        pic = np.asarray(Image.open(res))
        logger.info('pic shape %s' % (pic.shape,))
        logger.info('pic.flags.writeable %s' % (pic.flags.writeable,))
        pic = resize_and_pad(pic, 224, 224)
        pic = pic[:,:,0:3].copy()
        img = transform(pic)
        #logger.info('img size %s' % (img.size(),))
        return jsonify(model.inference(img))

    except Exception as e:
        logger.info(traceback.format_exc())
        return jsonify("an error occured")


