'''
Train and Eval Functions
'''
import config
from tqdm import tqdm
import utils
import dataset
import tensorflow as tf

def train_fn(model,data):
    """ Train the model with the given data and returns it.
    """
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # es=EarlyStopping(monitor='val_acc', baseline=0.85, patience=10 ,verbose=1,mode="max")
	# callbacks_list = [checkpoint]

    data=tf.data.Dataset.from_generator(dataset.TweetDataset(data,config.TOKENIZER,config.MAX_LEN).gen,
            output_types=dataset.gen_str).batch(config.TRAIN_BATCH_SIZE)

    model.fit(dataset,epochs=config.EPOCHS)
    return model

def eval_fn(model,dataset):
    """Eval the eval dataset and returns the metric"""
   
    data=tf.data.Dataset.from_generator(dataset.TweetDataset(data,config.TOKENIZER,config.MAX_LEN).gen,
            output_types=dataset.gen_str).batch(config.VALID_BATCH_SIZE)

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
            
        return orig_texts,pred_texts
        
    score=0
    for i,(data,_) in tqdm(enumerate(data)):
        orig_text=data["orig"]
        ext_text=data["ext"]
        preds=model.predict(data)
        targets,pred_texts=get_text(orig_text,preds)
        score=score+utils.jaccard(pred_texts,targets) 
    score=sum(score)/len(score)

    print("Total jaccard score : ",score)