from transformers import pipeline
import pandas as pd


def text2emotion():
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", return_all_scores=True)
    print(pipe("I bet everything will work out in the end :)"))

def list_labels():
    df = pd.read_csv("/home/u01230/SoccerNarration/train.csv")
    unique_labels = df["label"].drop_duplicates()
    print(unique_labels.tolist())
    with open("/home/u01230/SoccerNarration/unique_labels.txt", "w") as f:
        for label in unique_labels:
            f.write(f"{label}\n")

if __name__ == "__main__":
    list_labels()