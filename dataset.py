"""
Generator: Return -> (data(input_ids,attention_mask,token_type_ids),Sentiment(vector),Start_vector,End_vector)
"""
import config
import tensorflow as tf
import transformers

gen_str=({"orig":tf.string,"ext":tf.string,"input_ids":tf.int64,
                   "token_type_ids":tf.int64,"attention_mask":tf.int64,"lab":tf.int64},
                  tf.int64)

class TweetDataset:
    def __init__(self,data):
        self.tokenizer=config.TOKENIZER
        self.data=data
    self.max_len=config.MAX_LEN
        
    def get_target(self,data):
        start=[]
        end=[]
        text=data["text"]
        ext=data["selected_text"]
        sentiment=data["sentiment"]
        placed_txt=[0]*len(text)
    
        for i in range(len(text)):
            if text[i]==ext[0]:
                if(text[i:i+len(ext)]==ext):
                    placed_txt[i:i+len(ext)]=[1]*len(ext)
                    break
        encoded_text=self.tokenizer.encode(text)
        ids=encoded_text.ids
        type_ids=encoded_text.type_ids
        attention=encoded_text.attention_mask
        offsets=encoded_text.offsets

        start=[]
        for k,(i,j) in enumerate(offsets):
            if sum(placed_txt[i:j])>0:
                start.append(k)
      
        e=start[-1]
        s=start[0]
        if(len(ids)>self.max_len):
            ids=ids[:self.max_len-5]
            type_ids=type_ids[:self.max_len-5]
            attention=attention[:self.max_len-5]
            
        pad=self.max_len-len(ids)
        
        ids=ids+[0]*pad
        type_ids=type_ids+[0]*pad
        attention=attention+[0]*pad
        
        start=[[0]]*len(ids)
        end=[[0]]*len(ids)
        
        start[s]=[1]
        end[e]=[1]
        o=np.concatenate([start,end],axis=-1)
        return ({"orig":text,"ext":ext,"input_ids":ids,"token_type_ids":type_ids,"attention_mask":attention,"lab":[s,e]},o)
        

    def gen(self):
        """(inputs, targets)"""
        for i in range(len(self.data)):
            yield self.get_target(self.data.iloc[i])

    