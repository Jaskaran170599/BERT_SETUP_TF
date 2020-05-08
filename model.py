import tensorflow as tf
from transformers import *
import config

def get_model():

    ids=tf.keras.layers.Input(shape=(128,),name="input_ids",dtype=tf.int64)
    token_ids=tf.keras.layers.Input(shape=(128,),name="token_type_ids",dtype=tf.int64)    
    att_mask=tf.keras.layers.Input(shape=(128,),name="attention_mask",dtype=tf.int64)
    
    BL=TFBertModel.from_pretrained(config.BERT_PATH)
    L1,_=BL(ids,attention_mask=att_mask,token_type_ids=token_ids)

    drop=tf.keras.layers.Dropout(0.3)(L1)

    start=tf.keras.layers.Dense(2,activation="sigmoid",name="start")(L1)

    model=tf.keras.models.Model(inputs=[ids,token_ids,att_mask],outputs=[start]) 
    model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5))
    
    return model
