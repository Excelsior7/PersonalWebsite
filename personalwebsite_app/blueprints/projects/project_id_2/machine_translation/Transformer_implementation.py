import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import random
import math
import collections
import pickle
import time
from flask import current_app


# USER INPUT STRING MANIPULATION

def standardizeString(string, is_string_target):

    def _standardizeString(string):
        special_characters = '«»&~"#\'{([-|`_\\^@)]=}+¨£$¤%µ,?;.:!§*<>';
        numbers = '0123456789';
            
        len_string, _string = len(string), '';
        for i, char in enumerate(string):
            
            ## Handle special characters and numbers
            if char in special_characters or char in numbers:
                left_space, right_space = '', '';
                
                if i > 0 and string[i-1] != ' ':
                    left_space = ' ';

                if i+1 < len_string and string[i+1] != ' ' and string[i+1] not in special_characters and string[i+1] not in numbers:
                    right_space = ' ';
                
                _string += left_space + char + right_space;

            else:
                _string += char;

        return _string.lower().split(' ');

        
    ## Remove space characters
    space_characters = ['\u202f', '\u2009','\xa0'];
    for i in range(len(space_characters)):
        if space_characters[i] in string:
            string = string.replace(space_characters[i], ' ');

    _string = [s.strip() for s in string.split(' ') if s.strip() != ''];

    output = [];
    for i, _str in enumerate(_string):
        output += _standardizeString(_str);
        if i+1 < len(_string):
            output += ['<space>'];

    if is_string_target:
        output = ['<bos>'] + output;

    return output + ['<eos>'];    


def dataLoader(batch_size, shuffle, *tensors):
    TD = torch.utils.data.TensorDataset(*tensors);
    return torch.utils.data.DataLoader(TD, batch_size, shuffle);


def sequencesLen(dataset_examples):
    sequences_len = [];
    
    for i in range(len(dataset_examples)):
        sequences_len.append(len(dataset_examples[i]));
        
    return torch.tensor(sequences_len);

# VOCAB

class Vocab:
    def __init__(self, dataset2d):
        self.token_to_idx = {};
        self.idx_to_token = [];
        self.initVocab(dataset2d);
        
    def initVocab(self, dataset2d):
        token_freq = collections.Counter(
            [dataset2d[i][j] for i in range(len(dataset2d)) for j in range(len(dataset2d[i]))]);
        token_freq = token_freq.most_common();
  
        for i in range(len(token_freq)):
            self.token_to_idx[token_freq[i][0]] = i;
            self.idx_to_token.append(token_freq[i][0]);
             
    def tokenToIdx(self, dataset2d):
        for i in range(len(dataset2d)):
            dataset2d_irow = [];
            for j in range(len(dataset2d[i])):
                current_token = dataset2d[i][j];

                if current_token not in self.idx_to_token:
                    dataset2d_irow.append(self.token_to_idx['<special_begin>'])
                    for token in current_token:
                        if token not in self.idx_to_token:
                            dataset2d_irow.append(self.token_to_idx['<ukn>']);
                        else:
                            dataset2d_irow.append(self.token_to_idx[token]);
                    dataset2d_irow.append(self.token_to_idx['<special_end>'])
                else:
                    dataset2d_irow.append(self.token_to_idx[current_token]);

            dataset2d[i] = dataset2d_irow;
                    
        return torch.tensor(dataset2d);

    def idxToToken(self, dataset2d):
        dataset2d = dataset2d.tolist();
        
        for i in range(len(dataset2d)):
            for j in range(len(dataset2d[i])):
                dataset2d[i][j] = self.idx_to_token[dataset2d[i][j]];
        return dataset2d;

    def expandVocab(self, dataset2d):
        token_freq = collections.Counter(
            [dataset2d[i][j] for i in range(len(dataset2d)) for j in range(len(dataset2d[i]))]);
        token_freq = token_freq.most_common();
  
        for i in range(len(token_freq)):
            if token_freq[i][0] not in self.idx_to_token:
                self.token_to_idx[token_freq[i][0]] = len(self.idx_to_token);
                self.idx_to_token.append(token_freq[i][0]);
    
    def __len__(self):
        return len(self.idx_to_token);

# MODEL

def maskedSoftmax(QK, source_seq_len, mask):
    # QK.shape = (batch_size, num_steps, num_steps)
    
    QK_shape = QK.shape;
    
    if mask is True:
        mask_to_apply = ~(torch.arange(0,QK_shape[1])[None,:] < torch.arange(1,QK_shape[1]+1)[:,None]);
        mask_to_apply = mask_to_apply.unsqueeze(dim=0).repeat(QK_shape[0],1,1);
        
        QK[mask_to_apply] = -1e6;
    
    if source_seq_len is not None:
        steps = torch.arange(1, QK_shape[1]+1).unsqueeze(dim=0).repeat(QK_shape[1],1).unsqueeze(dim=0).repeat(QK_shape[0], 1, 1);
        valid_len = source_seq_len.unsqueeze(dim=1).unsqueeze(dim=1).repeat_interleave(repeats=QK_shape[1], dim=1);
        padding_mask = steps > valid_len;
        
        QK[padding_mask] = -1e6;
    
    return nn.functional.softmax(QK, dim=-1);

def scaledDotProductAttention(Q, K, V, dk, source_seq_len, mask):
    QK = torch.bmm(Q,K.transpose(1,2)) / math.sqrt(dk);
    
    return torch.bmm(maskedSoftmax(QK, source_seq_len, mask), V);

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dk, dv, dmodel):
        super().__init__();
        
        self.num_heads = num_heads;
        self.dk = dk;
        
        self.weights_params = nn.ModuleList();
        for i in range(num_heads):
            WQi = nn.Linear(dmodel,dk);
            WKi = nn.Linear(dmodel,dk);
            WVi = nn.Linear(dmodel,dv);
            
            weights = nn.ModuleList([WQi,WKi,WVi]);
            self.weights_params.append(weights);
            
        self.WO = nn.Linear(num_heads*dv,dmodel);
    
    def forward(self, queries, keys, values, source_seq_len=None, mask=False):
        # (queries|keys|values).shape = (batch_size, num_steps, dmodel)
        
        heads = [];
        
        for i in range(self.num_heads):
            WQi, WKi, WVi = self.weights_params[i];
            
            # ith head shape = (batch_size, num_steps, dv)
            heads.append(
                scaledDotProductAttention(WQi(queries), WKi(keys), WVi(values), self.dk, source_seq_len, mask));
        
        # heads.shape = (batch_size, num_steps, num_heads*dv)
        heads = torch.cat(heads, dim=-1);
        
        return self.WO(heads);

class FFN(nn.Module):
    def __init__(self, dmodel, dff):
        super().__init__();
        
        self.W1 = nn.Linear(dmodel,dff);
        self.W2 = nn.Linear(dff,dmodel);
        self.relu = nn.ReLU();
        
    def forward(self, X):
        # X.shape = (batch_size, num_steps, dmodel)
        
        return self.W2(self.relu(self.W1(X)));

class AddandNorm(nn.Module):
    def __init__(self, dmodel, dropout=0):
        super().__init__();
        
        self.LN = nn.LayerNorm(dmodel);
        self.dropout = nn.Dropout(dropout);
    
    def forward(self, X, Y):
        # (X|Y).shape = (batch_size, num_steps, dmodel)

        return self.LN(X + self.dropout(Y));

class PositionalEncoding(nn.Module):
    def __init__(self, dmodel, dropout, max_seq_len=1000):
        super().__init__();
        
        # It is possible that dmodel is odd but this makes the code more complex without adding value.
        assert dmodel % 2 == 0, "dmodel must be even";
        
        self.dropout = nn.Dropout();
        
        # t.shape = (max_seq_len, dmodel/2)
        t = torch.arange(0,max_seq_len).unsqueeze(dim=1).repeat_interleave(repeats=int(dmodel/2),dim=1);
        # w.shape = (max_seq_len, dmodel/2)
        wk = 1/torch.pow(10000, torch.arange(0,dmodel,step=2)/dmodel).unsqueeze(dim=0);
        wk = wk.repeat_interleave(repeats=max_seq_len,dim=0);
        
        # pos_encoding.shape = (max_seq_len, dmodel)
        self.pos_encoding = torch.zeros(max_seq_len, dmodel);
        self.pos_encoding[:,0::2] = torch.sin(wk*t);
        self.pos_encoding[:,1::2] = torch.cos(wk*t);

    def forward(self, X):
        # X.shape = (batch_size, num_steps, dmodel)
        X_shape = X.shape;
        
        pos_encoding = self.pos_encoding[:X_shape[1],:].unsqueeze(dim=0).repeat(X.shape[0],1,1);
        
        return self.dropout(pos_encoding + X);

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dmodel, dk, dv, dff, dropout):
        super().__init__();
        
        self.MHA = MultiHeadAttention(num_heads, dk, dv, dmodel);
        self.AAN = AddandNorm(dmodel, dropout);
        self.FFN = FFN(dmodel, dff);

    def forward(self, X, source_seq_len):
        # sli_out.shape = (batch_size, number of steps in src_X, dmodel)
        # sli stands for the ith sublayer of the encoder block.
        
        sl1_out = self.MHA(X, X, X, source_seq_len);
        sl1_out = self.AAN(X, sl1_out);
        
        sl2_out = self.FFN(sl1_out)
        sl2_out = self.AAN(sl1_out, sl2_out);
        
        return sl2_out;

class Encoder(nn.Module):
    def __init__(self, num_blocks, vocab_size, num_heads, dmodel, dk, dv, dff, dropout, max_seq_len=1000):
        super().__init__();
        
        self.num_blocks = num_blocks;
        self.embedding = nn.Embedding(vocab_size, dmodel);
        self.pencoding = PositionalEncoding(dmodel, dropout, max_seq_len);
        
        self.encoder_blocks = nn.ModuleList();
        for i in range(num_blocks):
            self.encoder_blocks.append(EncoderBlock(num_heads, dmodel, dk, dv, dff, dropout));

    def forward(self, src_X, source_seq_len_train):
        # X.shape = (batch_size, number of steps in src_X, dmodel)
        X = self.pencoding(self.embedding(src_X));
            
        for i in range(self.num_blocks):
            X = self.encoder_blocks[i](X, source_seq_len_train);
            
        return X;

class DecoderBlock(nn.Module):
    def __init__(self, num_heads, dmodel, dk, dv, dff, dropout):
        super().__init__();
        
        self.MHA1 = MultiHeadAttention(num_heads, dk, dv, dmodel);
        self.MHA2 = MultiHeadAttention(num_heads, dk, dv, dmodel);
        self.AAN = AddandNorm(dmodel, dropout);
        self.FFN = FFN(dmodel, dff);
        
    def forward(self, X, enc_output, mask=False):
        # X.shape = (batch_size, number of steps in bos_X, dmodel)
        # enc_output.shape = (batch_size, number of steps in src_X, dmodel)
        
        # sli_out.shape = (batch_size, number of steps in bos_X, dmodel)
        # sli stands for the ith sublayer of the decoder block.
        
        sl1_out = self.MHA1(X, X, X, None, mask);
        sl1_out = self.AAN(X, sl1_out);
        
        sl2_out = self.MHA2(sl1_out, enc_output, enc_output);
        sl2_out = self.AAN(sl1_out, sl2_out);
        
        sl3_out = self.FFN(sl2_out);
        sl3_out = self.AAN(sl2_out, sl3_out);
        
        return sl3_out; 

class Decoder(nn.Module):
    def __init__(self, num_blocks, vocab_size, num_heads, dmodel, dk, dv, dff, dropout, max_seq_len=1000):
        super().__init__();
        
        self.num_blocks = num_blocks;
        self.embedding = nn.Embedding(vocab_size, dmodel);
        self.pencoding = PositionalEncoding(dmodel, dropout, max_seq_len);
        
        self.W_out = nn.Linear(dmodel, vocab_size);
        
        self.decoder_blocks = nn.ModuleList();
        for i in range(num_blocks):
            self.decoder_blocks.append(DecoderBlock(num_heads, dmodel, dk, dv, dff, dropout));
        
    def forward(self, bos_X, enc_output):
        
        # X.shape = (batch_size, number of steps in bos_X, dmodel)
        X = self.pencoding(self.embedding(bos_X));
        
        mask = True if self.training else False;
        
        for i in range(self.num_blocks):
            X = self.decoder_blocks[i](X, enc_output, mask);
            
        return self.W_out(X);

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__();
        self.encoder = encoder;
        self.decoder = decoder;
        
    def forward(self, src_X, bos_X, source_seq_len):
        # src_X.shape = (batch_size, number of steps in src_X)
        # bos_X.shape = (batch_size, number of steps in bos_X)
        
        enc_output = self.encoder(src_X, source_seq_len);
        Y_hat = self.decoder(bos_X, enc_output);
        
        return Y_hat;
    
# INSTANTIATION OF MODEL AND HYPERPARAMETERS

## -- ENGLISH -> FRENCH IMPLEMENTATION -- ##
def loadVocab():
    with current_app.open_resource("blueprints/projects/project_id_2/machine_translation/saved_objects/vocabs_en_to_fr.pkl") as f:
        en_source_vocab = pickle.load(f);
        fr_target_vocab = pickle.load(f);

    return en_source_vocab, fr_target_vocab;

def loadModel(load_parameters=False, load_on_cpu=True):

    source_vocab, target_vocab = loadVocab();

    source_vocab_size = len(source_vocab);
    target_vocab_size = len(target_vocab);

    # SHARED HYPERPARAMETERS
    num_blocks = 6;
    num_heads = 8;
    dmodel, dk, dv, dff = 512, 64, 64, 2048;
    dropout = 0.1;

    encoder = Encoder(num_blocks, source_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
    decoder = Decoder(num_blocks, target_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
    model = EncoderDecoder(encoder, decoder);

    if load_parameters:
        with current_app.open_resource("blueprints/projects/project_id_2/machine_translation/saved_objects/parameters_Transformer_en_to_fr.tar") as f:
            if load_on_cpu:
                checkpoint = torch.load(f, map_location=torch.device('cpu'));
            else:
                checkpoint = torch.load(f);

        model.load_state_dict(checkpoint['model_state_dict']);

    if load_on_cpu == False:
        model.to(torch.device('cuda'));

    return model, source_vocab, target_vocab;


## -- ENGLISH <-> FRENCH IMPLEMENTATION -- ##
#
# def loadVocab(en_to_fr):
#     if en_to_fr:
#         with open("./saved_objects/vocabs_en_to_fr.pkl", 'rb') as f:
#             en_source_vocab = pickle.load(f);
#             fr_target_vocab = pickle.load(f);

#         return en_source_vocab, fr_target_vocab;

#     else:
#         with open("./saved_objects/vocabs_fr_to_en.pkl", 'rb') as f:
#             fr_source_vocab = pickle.load(f);
#             en_target_vocab = pickle.load(f);

#         return fr_source_vocab, en_target_vocab;

# def loadModel(en_to_fr, load_parameters=False, load_on_cpu=True):

#     source_vocab, target_vocab = loadVocab(en_to_fr);

#     source_vocab_size = len(source_vocab);
#     target_vocab_size = len(target_vocab);

#     # SHARED HYPERPARAMETERS
#     num_blocks = 2;
#     num_heads = 6;
#     dmodel, dk, dv, dff = 128, 32, 32, 512;
#     dropout = 0.1;

#     encoder = Encoder(num_blocks, source_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
#     decoder = Decoder(num_blocks, target_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
#     model = EncoderDecoder(encoder, decoder);

#     if load_parameters:
#         if load_on_cpu:
#             if en_to_fr:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_en_to_fr.tar', map_location=torch.device('cpu'));
#             else:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_fr_to_en.tar', map_location=torch.device('cpu'));
#         else:
#             if en_to_fr:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_en_to_fr.tar');
#             else:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_fr_to_en.tar');

#         model.load_state_dict(checkpoint['model_state_dict']);

#     if load_on_cpu == False:
#         model.to(torch.device('cuda'));

#     return model, source_vocab, target_vocab;


# PREDICTION GIVEN USER INPUT

def prediction(model,datasets,source_vocab,target_vocab):
    
    bos_idx = target_vocab.token_to_idx['<bos>'];
    eos_idx = target_vocab.token_to_idx['<eos>'];

    preds_outputs_src = [];
    preds_outputs_y = [];
    
    src_X, source_seq_len_test, Y = next(iter(datasets));    
    bos_X = torch.empty((len(src_X),1)).fill_(bos_idx).type(torch.int32);
    
    start = time.time();
    while(len(src_X) > 0):

        Y_hat = torch.transpose(model(src_X, bos_X, source_seq_len_test),0,1)[-1];
        preds = torch.argmax(Y_hat,dim=-1,keepdim=True);

        bos_X = torch.cat((bos_X,preds),dim=-1);

        ## Halt prediction if <eos> token.
        preds_is_eos = (preds == eos_idx).flatten();

        src_X_halt = source_vocab.idxToToken(src_X[preds_is_eos]);
        for i in range(len(src_X_halt)):
            preds_outputs_src.append(src_X_halt[i]);

        bos_X_halt = target_vocab.idxToToken(bos_X[preds_is_eos]);
        for i in range(len(bos_X_halt)):
            preds_outputs_y.append(bos_X_halt[i]);

        ## Delete terminated predictions.
        src_X = src_X[~preds_is_eos];
        bos_X = bos_X[~preds_is_eos];
        source_seq_len_test = source_seq_len_test[~preds_is_eos];

        if (time.time() - start) > 30:
            preds_outputs_y = None;
            break; 
         
    return preds_outputs_src, preds_outputs_y;

def standardizeOutput(output):
    if output is None:
        return "EXECUTION ERROR ON THE SERVER";
    else:
        output = output[0];

        standardized_ouput = "";
        output = output[1:-1];

        len_output = len(output);
        for i in range(len_output):
            if output[i] == '<special_begin>' or output[i] == '<special_end>':
                continue;
            elif output[i] == '<space>':
                standardized_ouput += " ";
            else:
                standardized_ouput += output[i];

        return standardized_ouput;

def translateUserInput(user_input, model, source_vocab, target_vocab):

    user_input_standardized = [standardizeString(user_input, False)];
    user_input_standardized_tokenized = source_vocab.tokenToIdx(user_input_standardized);

    user_input_sequence_len = sequencesLen(user_input_standardized);

    # torch.tensor([[0]]) is just here to replace the Y (i.e. the traduction of the user_input) that I don't have access to in production.
    data = dataLoader(1, False, user_input_standardized_tokenized, user_input_sequence_len, torch.tensor([[0]]));

    _, out_y = prediction(model, data, source_vocab, target_vocab);

    return standardizeOutput(out_y);


## CREATE VOCABULARIES ##
#
## Why do we need to recreate vocabularies pickle objects from here ?
## (see option 2: https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules)
## (note: option 1 from the same link forced me to modify a file in the virtual environment)

def createVocabs(en_to_fr):
    with current_app.open_resource("blueprints/projects/project_id_2/machine_translation/data/en_fra.txt", "r") as f:
        examples = f.readlines();

    en_examples, fr_examples = standardizeExamples(examples, en_to_fr);

    augmentData(en_examples, fr_examples);

    if en_to_fr:
        _, source_vocab, target_vocab = datasets(en_examples, fr_examples, [5,10,15,20,25], 1, 1024);

    else:
        _, source_vocab, target_vocab = datasets(fr_examples, en_examples, [5,10,15,20,25], 1, 1024);

    dumpVocab(source_vocab, target_vocab, en_to_fr);


def standardizeExamples(examples, en_to_fr):

    en_examples, fr_examples = [], [];

    for i in range(len(examples)):
        exi = examples[i][0:examples[i].find('CC-BY 2.0')];

        exi = exi.split('\t');

        if en_to_fr:
            en_examples.append(standardizeString(exi[0], False));
            fr_examples.append(standardizeString(exi[1], True));
        else:
            en_examples.append(standardizeString(exi[0], True));
            fr_examples.append(standardizeString(exi[1], False));

    return en_examples, fr_examples;


def specialEntries(i, token_noize_period, ukn_period, word_entry_point, word_replacement, symbols, en_examples, fr_examples):
    eni_augmented, fri_augmented = [], [];

    if i % token_noize_period == 0:
        tokens_noize = [];
        if word_replacement == None:
            for _ in range(len(word_entry_point)):
                tokens_noize.append(random.choice(symbols));
        else:
            for _ in range(len(word_replacement)):
                tokens_noize.append(random.choice(symbols));
    
    for t in en_examples[i]:
        if t == word_entry_point:
            if i % ukn_period == 0:
                eni_augmented.append('<ukn>');        
            else:
                eni_augmented.append('<special_begin>');
                if i % token_noize_period == 0:
                    for tn in tokens_noize:
                        eni_augmented.append(tn);
                else:
                    for token in word_replacement:
                        eni_augmented.append(token);
                eni_augmented.append('<special_end>');
        else:
            eni_augmented.append(t);
    en_examples.append(eni_augmented);
        
    for t in fr_examples[i]:
        if t == word_entry_point:
            if i % ukn_period == 0:
                fri_augmented.append('<ukn>'); 
            else:
                fri_augmented.append('<special_begin>');
                if i % token_noize_period == 0:
                    for tn in tokens_noize:
                        fri_augmented.append(tn);
                else:
                    for token in word_replacement:
                        fri_augmented.append(token);
                fri_augmented.append('<special_end>');
        else:
            fri_augmented.append(t);
    fr_examples.append(fri_augmented);

def augmentData(en_examples, fr_examples):

    with current_app.open_resource("blueprints/projects/project_id_2/machine_translation/data/babynames-clean.csv", "r") as f:
        names_dataset = pd.read_csv(f);

    boy_names = names_dataset[names_dataset.iloc[:,1] == "boy"].reset_index().iloc[:,1];
    len_boy_names = len(boy_names);
    girl_names = names_dataset[names_dataset.iloc[:,1] == "girl"].reset_index().iloc[:,1];
    len_girl_names = len(girl_names);
    
    counter_boy = 0;
    counter_girl = 0;
    duplicate_multiplier_boy = 2;
    duplicate_multiplier_girl = 2;

    symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    symbols += ['&', 'é', '~', '"', '#', '\'', '{', '(', '[', '-', '|', 'è', '`', '_', '\\', 'ç', '^', 'à', '@', ')', ']', '=', '°', '}', '+', '/', '*', '?', ',', ';', '.', ':', '!', '§', '¨', '%', 'ù', '$', '£', '¤', 'µ', '«', '»', '<', '>'];


    for i in range(len(en_examples)):
        eni = en_examples[i];

        if 'tom' in eni and 'tom' in fr_examples[i]:
            for j in range(duplicate_multiplier_boy):
                boy_name = boy_names.iloc[((counter_boy*duplicate_multiplier_boy)+j) % len_boy_names].lower();
                eni_augmented = [boy_name if t=='tom' else t for t in eni];
                fri_augmented = [boy_name if t=='tom' else t for t in fr_examples[i]];

                en_examples.append(eni_augmented);
                fr_examples.append(fri_augmented);
            
            counter_boy += 1;

            specialEntries(i, 
            token_noize_period=3, 
            ukn_period=20, 
            word_entry_point='tom', 
            word_replacement=boy_name ,
            symbols=symbols, 
            en_examples=en_examples, 
            fr_examples=fr_examples);

        if 'mary' in eni and ('mary' in fr_examples[i] or 'marie' in fr_examples[i]):
            for j in range(duplicate_multiplier_girl):
                girl_name = girl_names.iloc[((counter_girl*duplicate_multiplier_girl)+j) % len_girl_names].lower();
                eni_augmented = [girl_name if t=='mary' else t for t in eni];
                fri_augmented = [girl_name if (t=='mary'or t=='marie') else t for t in fr_examples[i]];

                en_examples.append(eni_augmented);
                fr_examples.append(fri_augmented);

            counter_girl += 1;

        if 'tom' in eni and 'mary' in eni:
            boy_name = random.choice(boy_names).lower();
            girl_name = random.choice(girl_names).lower();

            eni_augmented = [girl_name if t=='mary' else t for t in eni];
            eni_augmented = [boy_name if t=='tom' else t for t in eni_augmented];

            fri_augmented = [girl_name if (t=='mary'or t=='marie') else t for t in fr_examples[i]];
            fri_augmented = [boy_name if t=='tom' else t for t in fri_augmented];

            en_examples.append(eni_augmented);
            fr_examples.append(fri_augmented);


        en_fr_identical_words = ['paris', 'canada', 'facebook', 'boston', 'jupiter', 'pizza', 'vodka', 'piano', 'train', 'radio', 'message', 'danger'];
        duplicate_multiplier = 3;
        for word in en_fr_identical_words:
            if word in eni and word in fr_examples[i]:
                for _ in range(duplicate_multiplier):
                    specialEntries(i, 
                    token_noize_period=1, 
                    ukn_period=20, 
                    word_entry_point=word, 
                    word_replacement=None, 
                    symbols=symbols,
                    en_examples=en_examples, 
                    fr_examples=fr_examples);

def dataLoader(batch_size, shuffle, *tensors):
    TD = torch.utils.data.TensorDataset(*tensors);
    return torch.utils.data.DataLoader(TD, batch_size, shuffle);


## Determine the longest sequence among dataset_examples 
## and complete the other sequences with the <pad> token so that their length matches the longest.
def padding(dataset_examples):
    
    max_length = 0;

    def maxLength(dataset, max_length):
        for i in range(len(dataset)):
            if len(dataset[i]) > max_length:
                max_length = len(dataset[i]);
        return max_length;
                
    max_length = maxLength(dataset_examples, max_length);
    
    def pad(dataset, max_length):
        for i in range(len(dataset)):
            if len(dataset[i]) < max_length:
                dataset[i] += ['<pad>']*(max_length-len(dataset[i]));
        return dataset;
    
    dataset_examples = pad(dataset_examples, max_length);
                
    return dataset_examples;


# groups is a list of int, it is the list that determines the different dataset groups according to the sequences length;
# e.g. if groups = [5,10,15,20,25] then the following groups will be made:
# (0,5], (5,10], (10, 15], (15, 20], (20, 25].
#
# Assumptions: groups = [g1,g2,g3,...,gG];
# g1,g2,g3,...,gG > 0;
# g1<g2<g3<...<gG;

# The group_maker can takes the value 1 or 2 and is the parameter that determines
# from which datasets: source_examples (1) or target_examples (2) we make the groups. 

def datasets(source_examples, target_examples, groups, group_maker, batch_size_train):

    source_examples_groups, target_examples_groups = [], [];

    # CHECKS GROUPS ASSUMPTIONS
    if min(groups) < 0:
        raise ValueError("groups elements must be positive");
    groups.sort();

    if group_maker == 1:
        examples_len = sequencesLen(source_examples);
    else:
        examples_len = sequencesLen(target_examples);

    # CREATE GROUPS
    for i in range(len(groups)):
        group_lower_bound = 0 if i == 0 else groups[i-1];
        group_upper_bound = groups[i];

        lower_bound_true = group_lower_bound < examples_len;
        upper_bound_true = examples_len <= group_upper_bound;

        lower_upper_bound_true_indices = (lower_bound_true & upper_bound_true).nonzero(as_tuple=True)[0];

        source_examples_group = list(source_examples[i] for i in lower_upper_bound_true_indices);
        target_examples_group = list(target_examples[i] for i in lower_upper_bound_true_indices);

        source_examples_groups.append(source_examples_group);
        target_examples_groups.append(target_examples_group);

    number_groups = len(source_examples_groups);


    # SOURCE SEQUENCES LENGTH
    source_seq_len = [];
    for i in range(number_groups):
        source_seq_len.append(sequencesLen(source_examples_groups[i]));

    # PADDING 
    for i in range(number_groups):
        source_examples_groups[i] = padding(source_examples_groups[i]);
        target_examples_groups[i] = padding(target_examples_groups[i]);


    # CREATE VOCAB
    for i in range(number_groups):
        if i == 0:
            source_vocab = Vocab(source_examples_groups[i])
            target_vocab = Vocab(target_examples_groups[i]);
        else:
            source_vocab.expandVocab(source_examples_groups[i]);
            target_vocab.expandVocab(target_examples_groups[i]);

    # TOKEN TO INDEX
    for i in range(number_groups):
        source_examples_groups[i] = source_vocab.tokenToIdx(source_examples_groups[i]);
        target_examples_groups[i] = target_vocab.tokenToIdx(target_examples_groups[i]);

    
    # TRAIN DATASETS
    datasets_train = [];

    for i in range(number_groups):
        if len(source_examples_groups[i]) == 0:
            continue;

        src_train = source_examples_groups[i];
        src_seq_len_train = source_seq_len[i];

        trg_train_in = target_examples_groups[i][:,:-1];
        trg_train_out = target_examples_groups[i][:,1:];

        datasets_train.append(dataLoader(batch_size_train, True, src_train, src_seq_len_train, trg_train_in, trg_train_out));

    
    return datasets_train, source_vocab, target_vocab;



def dumpVocab(source_vocab, target_vocab, en_to_fr):
    if en_to_fr:
        with open("/home/excelsior/Desktop/PersonalWebsite/personalwebsite_app/blueprints/projects/project_id_2/machine_translation/saved_objects/vocabs_en_to_fr.pkl", "wb") as f:
            pickle.dump(source_vocab, f, pickle.HIGHEST_PROTOCOL);
            pickle.dump(target_vocab, f, pickle.HIGHEST_PROTOCOL);
    else:
        with open("/home/excelsior/Desktop/PersonalWebsite/personalwebsite_app/blueprints/projects/project_id_2/machine_translation/saved_objects/vocabs_fr_to_en.pkl", "wb") as f:
            pickle.dump(source_vocab, f, pickle.HIGHEST_PROTOCOL);
            pickle.dump(target_vocab, f, pickle.HIGHEST_PROTOCOL);
