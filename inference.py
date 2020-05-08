import tensorflow as tf
import config
from tqdm import tqdm



def get_target(self,data):
    text=data["text"]
    text_id=data["TEXTID"]
    sentiment=data["sentiment"]
    
    encoded_text=config.TOKENIZER.encode(text)
    ids=encoded_text.ids
    type_ids=encoded_text.type_ids
    attention=encoded_text.attention_mask
    
    if(len(ids)>config.MAX_LEN):
        ids=ids[:config.MAX_LEN-5]
        type_ids=type_ids[:config.MAX_LEN-5]
        attention=attention[:config.MAX_LEN-5]
        
    pad=config.MAX_LEN-len(ids)

    ids=ids+[0]*pad
    type_ids=type_ids+[0]*pad
    attention=attention+[0]*pad

    return {"orig":text,"id":text_id,"input_ids":ids,"token_type_ids":type_ids,"attention_mask":attention}


def gen(data):
    """(inputs, targets)"""
    for i in range(len(data)):
        yield get_target(data.iloc[i])

def get_text(text,pred):
        
        pred_texts=[]
        orig_texts=[]
        text=text.numpy()
        pred=tf.argmax(pred,axis=1).numpy()

        for t,p in zip(text,pred):
            orig_texts.append(t.decode("utf-8"))
            t=config.TOKENIZER.encode(orig_texts[-1]).offsets
            i,j=p[0],p[1]
            pred_texts.append(orig_texts[-1][t[i][0]:t[j][1]])
            
        return pred_texts

def run():
    test_data=pd.read_csv(config.TESTING_FILE)
    test_dataset=tf.data.Dataset.from_generator(gen(test_data),
            output_types={"orig":tf.string,"id":tf.string,"input_ids":tf.int32,
            "token_type_ids":tf.int32,"attention_mask":tf.int32}
            ).batch(config.VALID_BATCH_SIZE)

    model=tf.keras.models.load_model(config.MODEL_PATH)

    output_texts=[]
    for data in tqdm(test_dataset):
        orig_text=data["orig"]
        preds=model.predict(data)
        output_texts=output_texts+get_text(orig_text,preds)
    
    sample = pd.read_csv("kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
    sample.loc[:, 'selected_text'] = output_texts
    sample.to_csv("submission.csv", index=False)

if __name__=="__main__":
    run()
