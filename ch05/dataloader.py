
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
       
def create_dataloader_v1(text, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # create dataset
    dataset =  GPTDatasetV1(text, tokenizer, max_length, stride)

    # create dataloder
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader 

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("gpt2")
    raw_text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
         "of someunknownPlace."
    )
    encoded_text = tokenizer.encode(raw_text, allowed_special={'<|endoftext|>'})
    print(raw_text[:10])
    print(encoded_text[:10])
    print(tokenizer.encode('some'))
    print(tokenizer.encode('unknown'))
    print(tokenizer.encode('Place'))
    print(tokenizer.encode('someunknownPlace.'))
   
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    context_length = 1024
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    max_length = 8
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=2)
    print (dataloader)

    for batch in dataloader:
        x, y = batch
        print('x: ', x, 'y:', y)
        token_emb = token_embedding_layer(x)
        position_emb = pos_embedding_layer(torch.arange(max_length))

        input_emb = token_emb + position_emb
        print('input_emb: ', input_emb)
