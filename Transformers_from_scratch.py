import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads) :
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert(self.head_dim*heads == embed_size), "Embed size needs to be divisible by heads"

        # compute the values, keys and queries for all heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0] # how many examples we send in at a same time (batch_size)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # depending of where we use the attention_mechanism
        # they corresponding to the source sentences length and the target sentences length (on ne connaît pas à priori la taille des phrases)  

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # multiply the values with the keys and call the results energy
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape : (N, queries_len, heads, heads_dim:=d)
        # keys shape : (N, keys_len, heads, heads_dim)
        # queries shape : (N, query_len, heads, heads_dim)
        # energy : Q.T*K (N, heads, query_len, key_len). The query_len is the target source sentences, the key_len is the source sentences
        # energy operations means that for each word in the target source sentences, how much do we pay attention to each word in the source sentence
         
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e12")) 

        attention = torch.softmax(energy/ (self.embed_size ** 0.5), dim = 3) # normalize accross the key_len

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape : (N, heads, query_len, key_len)
        # values shape : (N, value_len=key_len:=l, heads, heads_dim)
        # (N, query_len, heads, heads_dim)
        # reshape to concatenate all heads

        out = self.fc_out(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) :
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask): # mask in encoder for padding's tokens
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion,
                 dropout, max_length): # max_length because i'll use position embedding
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # word and positions embedding
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x)+ self.position_embedding(positions))

        # Transfomer block
        for layer in self.layers:
            out = layer(out, out, out, mask) # in the encoder, we have key=query=value

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, targert_mask): # the value and the key from the decoder
        attention = self.attention(x, x, x, targert_mask)
        query = self.dropout(self.norm(attention + x)) # the query from the attention on the output
        out = self.transformer_block(value, key, query, src_mask)
        return out # à clarifier (confusion possible sur l'origine des key, query et value )
    
class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, heads, device, forward_expansion,
                 dropout, max_length): # max_length because i'll use position embedding
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                                     for _ in range(num_layers)])

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        #self.softmax = nn.Softmax(2)

    def forward(self, x, enc_out, src_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x)+ self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask) # key=value

        out = self.fc_out(x)
        #out = self.softmax(out)
        return out 
    
class Transformer(nn.Module):
    def __init__(self, src_voc_size, target_voc_size, src_pad_idx, target_pad_idx,
                 embed_size = 512, num_layers = 6, forward_expansion =4, heads = 8,
                 dropout = 0, device = 'cpu', max_length = 100) :
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_voc_size, embed_size, num_layers, heads, device, forward_expansion,
                               dropout, max_length)
        self.decoder = Decoder(target_voc_size, embed_size, num_layers, heads, device, forward_expansion,
                               dropout, max_length)
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src!= self.src_pad_idx).unsqueeze(1).unsqueeze(2) # we don't want to consider the padding's tokens in the src sentence
        #shape : (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_target_mask(self, target):
        N, target_len = target.shape
        tgt_padding_mask = (target != self.target_pad_idx).unsqueeze(1).unsqueeze(2)
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(N,1, target_len, target_len).bool() # one mask for each training sentence
        target_mask = tgt_padding_mask & target_mask
        return target_mask.to(self.device)
    
    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)
        return out
    

# quick test
"""
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1,2,3,4,5,0], [1,2,3,4,5,6]]).to(device)
    target = torch.tensor([[1,8,13,4,5,0,0], [1,2,3,4,5,6,15]]).to(device)

    src_pad_idx = 0
    target_pad_idx = 0
    src_vocab_size = 7
    target_vocab_size = 16

    model = Transformer(src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx).to(device)
    out = model(x, target[:, :-1]) # predict end of sentence
    print(out.shape)
    print(out[1, :, :])
"""




    



