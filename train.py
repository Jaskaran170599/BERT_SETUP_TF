import pandas as pd
import config
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import engine
from model import get_model

def run():
    data=pd.read_csv(config.TRAINING_FILE).dropna()
    train,eval=train_test_split(data,random_state=1,test_size=0.2)

    model=get_model()
    model=engine.train_fn(model,train)
    score=engine.eval_fn(model,eval)

    print("EVAL Score : ",score)

    model.save_model(config.MODEL_PATH)


if __name__=="__main__":
    run()