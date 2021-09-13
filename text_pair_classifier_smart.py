import random
import os
os.environ['CUDA_VISIBLE_DEVICES']= '2'
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, BertLayer, BertForMaskedLM
from transformers import BertForSequenceClassification, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import argparse

class EmbAdapterDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data=data
        self.tokenizer=tokenizer
        self.max_length=args.max_length
        self.pad_token = 0
        self.mask_ids = 103

    def __getitem__(self, index):
        x1, x2, y = self.data[index]
        inputs = self.tokenizer.encode_plus(x1, x2, add_special_tokens=True, max_length=self.max_length,
                                            truncation=True)
        padding_length = self.max_length - len(inputs["input_ids"])
        input_ids = torch.tensor(inputs["input_ids"] + padding_length * [self.pad_token], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"] + padding_length * [0], dtype=torch.long)
        attention_mask = torch.tensor([1]*len(inputs["input_ids"])+padding_length*[0], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask, y

    def __len__(self):
        return len(self.data)

class FreeLB(nn.Module):
    def __init__(self, args, nclasses):
        super(FreeLB, self).__init__()
        config_atk = BertConfig.from_pretrained(args.mlm_path, num_labels=nclasses)
        self.tgt_model = BertForSequenceClassification.from_pretrained(args.mlm_path, config=config_atk).cuda()
        self.variance = args.variation
        self.variance_vda = args.variation_vda
        self.step = args.step
        self.criterion = nn.CrossEntropyLoss()
        self.kl_criterion = nn.KLDivLoss()

    def forward(self, input_ids, attention_masks, token_type_ids):
        return self.tgt_model(input_ids, attention_masks, token_type_ids)

    def output_with_emb(self, embedding_output, extended_attention_mask, head_mask, encoder_extended_attention_mask):
        encoder_outputs = self.tgt_model.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.tgt_model.bert.pooler(sequence_output) if self.tgt_model.bert.pooler is not None else None
        pooled_output = self.tgt_model.dropout(pooled_output)
        logits = self.tgt_model.classifier(pooled_output)
        return logits

    def adv_train(self, input_ids, attention_masks, token_type_ids, y, optim, scheduler, mlm_model):
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask: torch.Tensor = self.tgt_model.bert.get_extended_attention_mask(attention_masks,
                                                                                                input_shape, device)

        encoder_extended_attention_mask = None
        head_mask = self.tgt_model.bert.get_head_mask(None, self.tgt_model.bert.config.num_hidden_layers)
        embedding_output = self.tgt_model.bert.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
        )

        noise = torch.randn(embedding_output.size(), device=device) * self.variance
        noise = noise.cuda()
        noise.requires_grad = True

        n_embedding_output = embedding_output + noise
        n_logits = self.output_with_emb(n_embedding_output, extended_attention_mask, head_mask,
                                       encoder_extended_attention_mask)
        logits = self.output_with_emb(embedding_output, extended_attention_mask, head_mask,
                                       encoder_extended_attention_mask)

        n_loss = 0.5*self.kl_criterion(torch.log_softmax(logits, -1), torch.softmax(n_logits, -1))+\
                 0.5*self.kl_criterion(torch.log_softmax(n_logits, -1), torch.softmax(logits, -1))
        noise_grad = torch.autograd.grad(n_loss, noise)[0]
        noise = noise + (noise_grad / torch.norm(noise_grad)).mul_(1e-3)
        noise = torch.clamp(noise, -self.variance, self.variance)
        #update the noise finish

        n_embedding_output = embedding_output + noise
        n_logits = self.output_with_emb(n_embedding_output, extended_attention_mask, head_mask,
                                        encoder_extended_attention_mask)
        logits = self.output_with_emb(embedding_output, extended_attention_mask, head_mask,
                                      encoder_extended_attention_mask)
        n_loss = 0.5 * self.kl_criterion(torch.log_softmax(logits, -1), torch.softmax(n_logits, -1)) + \
                 0.5 * self.kl_criterion(torch.log_softmax(n_logits, -1), torch.softmax(logits, -1))
        loss = self.criterion(logits, y)

        probs = mlm_model(input_ids, attention_masks, token_type_ids)[0]
        probs = probs / torch.sum(probs, -1, keepdim=True)
        noise = torch.randn(probs.size(), device=device) * self.variance_vda
        probs = torch.softmax(probs + noise.cuda(), -1)  # [B, Len, V]
        word_embs = self.tgt_model.bert.embeddings.word_embeddings.weight
        input_embeds = torch.matmul(probs, word_embs)
        embedding_output = self.tgt_model.bert.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=input_embeds,
        )
        vda_logits = self.output_with_emb(embedding_output, extended_attention_mask, head_mask,
                                      encoder_extended_attention_mask)
        vda_loss = 0.5 * self.kl_criterion(torch.log_softmax(logits, -1), torch.softmax(vda_logits, -1)) + \
                 0.5 * self.kl_criterion(torch.log_softmax(vda_logits, -1), torch.softmax(logits, -1))

        total_loss=loss+n_loss+vda_loss
        total_loss.backward()
        optim.step()
        #print({name:param.grad for name, param in self.tgt_model.classifier.named_parameters()})
        scheduler.step()
        self.tgt_model.zero_grad()

        return loss.cpu().item(), n_loss.cpu().item(), vda_loss.cpu().item()

    def defense(self, input_ids, attention_mask, token_type_ids, mlm_model):
        output_attentions = self.tgt_model.bert.config.output_attentions
        output_hidden_states = (self.tgt_model.bert.config.output_hidden_states)

        input_shape = input_ids.size()

        device = input_ids.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.tgt_model.bert.get_extended_attention_mask(attention_mask,
                                                                                                input_shape, device)
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.tgt_model.bert.get_head_mask(None, self.tgt_model.bert.config.num_hidden_layers)

        probs = mlm_model(input_ids, attention_mask, token_type_ids)[0]
        probs = probs / torch.sum(probs, -1, keepdim=True)
        noise = torch.randn(probs.size()) * self.variance_vda

        probs = torch.softmax(probs + noise.cuda(), -1)  # [B, Len, V]

        word_embs = self.tgt_model.bert.embeddings.word_embeddings.weight
        input_embeds = torch.matmul(probs, word_embs)

        embedding_output = self.tgt_model.bert.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=input_embeds,
        )
        encoder_outputs = self.tgt_model.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]

        pooled_output = self.tgt_model.bert.pooler(sequence_output) if self.tgt_model.bert.pooler is not None else None

        # logits = self.output_linear(prompt_output[0])
        logits = self.tgt_model.classifier(pooled_output)

        return logits

def reader_data(filename, add=True):
    #return [[ori_text, adv_text, label], ... ,[]]
    f=open(filename, encoding='utf-8')
    output=[]
    for line in f:
        label, text1, text2 = line.strip().split('\t')
        label = int(label)
        output.append([text1, text2, label])
    return output


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qnli")
    parser.add_argument("--mlm_path", type=str, default="../bert_file", help="xxx mlm")
    parser.add_argument("--tgt_path", type=str, default="../TextFooler/target_models/mrpc",
                        help="xxx classifier")
    parser.add_argument("--save_path", type=str, default="saved/qnli_smart_vda.pt", help="xxx mlm")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_label", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_warmup", type=float, default=0.05)
    parser.add_argument("--variation", type=float, default=0.02)
    parser.add_argument("--variation_vda", type=float, default=0.05)
    parser.add_argument("--step", type=int, default=2)

    args = parser.parse_args()
    data_path = 'data/'+str(args.dataset)
    train_features = reader_data(os.path.join(data_path, 'train.txt'))
    dev_features = reader_data(os.path.join(data_path, 'dev.txt'))
    test_features = reader_data(os.path.join(data_path, 'test.txt'))

    num_label = args.num_label

    print('start process')

    # tokenizer_mlm = BertTokenizer.from_pretrained(mlm_path, do_lower_case=True)
    mlm_model = BertForMaskedLM.from_pretrained(args.mlm_path).cuda()
    tokenizer_tgt = BertTokenizer.from_pretrained(args.mlm_path, do_lower_case=True)

    train_data = EmbAdapterDataset(args, train_features, tokenizer_tgt)
    dev_data = EmbAdapterDataset(args, dev_features, tokenizer_tgt)
    test_data = EmbAdapterDataset(args, test_features, tokenizer_tgt)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    print('start building model')
    model = FreeLB(args, args.num_label)
    model= model.cuda()

    params=model.parameters()
    #need_grad = lambda x: x.requires_grad
    optimizer = AdamW(
        params,
        lr=args.lr, eps=1e-8, weight_decay=0.01,
    )
    total_num = len(train_data) // args.batch_size * args.max_epoch
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup * total_num,
                                                                   num_training_steps=total_num)
    best_ratio=0
    for epoch in range(args.max_epoch):
        model.train()
        attack_ratio = 0
        total_num = 0
        total_loss = []
        for batch in tqdm(train_dataloader):
            model.zero_grad()
            input_ids, token_type_ids, attention_mask, y = [ele.cuda() for ele in batch]
            #origin_emb = model.produce_emb(input_ids, attention_mask, token_type_ids)
            loss, kl_loss, vda_loss = model.adv_train(input_ids, attention_mask, token_type_ids, y, optimizer, scheduler, mlm_model)
            total_loss.append([loss, kl_loss, vda_loss])

        print('Epoch %d, the training ce loss is %f, the kl loss is %f, vda loss is %f, the total number is %f'
              % (epoch, sum([ele[0] for ele in total_loss])/len(total_loss),
                 sum([ele[1] for ele in total_loss])/len(total_loss), sum([ele[2] for ele in total_loss])/len(total_loss), total_num))

        attack_ratio = 0
        total_num = 0
        model.eval()
        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids, token_type_ids, attention_mask, y = [ele.cuda() for ele in batch]
                probs = model(input_ids, attention_mask, token_type_ids)[0]  # , token_type_ids)[0]
                argmax_probs = torch.argmax(probs, dim=-1)
                success_num = torch.sum((argmax_probs == y).float()).cpu().item()
                attack_ratio += success_num
                total_num += input_ids.size(0)
        if best_ratio<attack_ratio:
            torch.save(model.tgt_model.state_dict(), args.save_path)
            print('--------save once-----------')
            best_ratio=attack_ratio
        print('The dev set defense success attack ratio is %f, the total number is %f' % (attack_ratio / total_num, total_num))

        attack_ratio = 0
        total_num = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, token_type_ids, attention_mask, y = [ele.cuda() for ele in batch]
                probs = model(input_ids, attention_mask, token_type_ids)[0]  # , token_type_ids)[0]
                argmax_probs = torch.argmax(probs, dim=-1)
                success_num = torch.sum((argmax_probs == y).float()).cpu().item()
                attack_ratio += success_num
                total_num += input_ids.size(0)
        print('The test set defense success attack ratio is %f, the total number is %f' % (attack_ratio / total_num, total_num))

if __name__ == '__main__':
    run()
