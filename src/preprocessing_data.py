import os
import re
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

logger = logging.getLogger(__name__)

def read_data(data_dir):
    warmup_df = pd.read_json(os.path.join(data_dir, "ise-dsc01-warmup.json"), orient="index")
    train_df = pd.read_json(os.path.join(data_dir, "ise-dsc01-train.json"), orient="index")
    df = pd.concat([warmup_df, train_df], axis=0)
    df.insert(loc=0, column="id", value=df.index.values)
    df.reset_index(inplace=True, drop=True)
    return df


def split_into_sentences(text: str) -> list[str]:
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","<prd>”. ")
    if "\"" in text: text = text.replace(".\"","<prd>\". ")
    # if "!" in text: text = text.replace("!\"","<ellipsis>\". ")
    # if "?" in text: text = text.replace("?\"","<qm>\". ")
    text = text.replace(". ",".<stop>")
    # text = text.replace("?","?<stop>")
    # text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    # text = text.replace("<ellipsis>","!")
    # text = text.replace("<qm>","?")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    # sentences = re.split(r'(?<=[.!?…])\s+', text)
    return sentences

def preprocess_text(text: str) -> str:
    if text is not None:
        # text = re.sub(r"['\",\.\?:\-!]", "", text)
        text = text.strip()
        text = " ".join(text.split())
        # text = text.lower()
        return text
    return None

def evidence_isIn_sentence (row):
    if (row['evidence'] is not None):
        if row['evidence'] in row['sentence']:
            return True
        else:
            return False
    return False

def process_data(df):
    df.insert(loc=2, column="sentence", value=df.context.apply(split_into_sentences))
    df.insert(loc=1, column="num_sentence", value=df.sentence.apply(lambda x: np.arange(len(x))))
    df = df.explode(column=["sentence", "num_sentence"])
    df.sentence = df.sentence.apply(preprocess_text)
    df.evidence = df.evidence.apply(preprocess_text)
    df.insert(loc=0, column="id_sentence", value=df.id.apply(str)+"_"+df.num_sentence.apply(str))
    df.reset_index(inplace=True, drop=True)
    return df

def create_train_pos(df):
    df = df[df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df.reset_index(inplace=True, drop=True)
    df = df.loc[:,["id_sentence", "sentence", "claim", "verdict", "domain"]]
    return df

def create_train_neg(df):
    df = df[~df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df.loc[:, "verdict"] = "NEI"
    df.reset_index(inplace=True, drop=True)
    df = df.loc[:,["id_sentence", "sentence", "claim", "verdict", "domain"]]
    return df

def create_dev(df):
    dev_neg_df = df[~df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df.loc[df.id_sentence.isin(dev_neg_df.id_sentence),"verdict"] = "NEI"
    return df.loc[:,["id_sentence", "id", "num_sentence", "sentence", "claim", "domain", "verdict"]]


def write_data(output_dir, train_pos_df, train_neg_df, dev_df):
    train_pos_df.to_json(os.path.join(output_dir, "train_pos_sentences.json"), force_ascii=False, indent=4, orient='index')
    train_neg_df.to_json(os.path.join(output_dir, "train_neg_sentences.json"), force_ascii=False, indent=4,orient='index')
    dev_df.to_json(os.path.join(output_dir, "dev_sentences.json"), force_ascii=False, indent=4,orient='index')


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="./datasets/",
                        type=str,
                        help="The input data dir (warmup data).")
    parser.add_argument("--output_dir",
                        default="./datasets/",
                        type=str,
                        help="The output data dir where data has been converted to fit for running the model.")
    parser.add_argument("--data_splitting",
                    default=False,
                    # action='store_true',
                    help="Data splitting. True: train set, test set and validation set. False: train set, validation set")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
    # Raise a warning or display a message
        print("Warning: The specified data directory does not exist.")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info("******Convert data******")
    df = read_data(args.data_dir)

    train_df, val_df = train_test_split(df, train_size=0.7, random_state=42)
    if args.data_splitting:
        val_df, test_df = train_test_split(val_df, train_size=0.5, random_state=42)
    train_df = process_data(train_df)
    train_pos_df = create_train_pos(train_df)
    train_neg_df = create_train_neg(train_df)

    val_df = process_data(val_df)
    val_df = create_dev(val_df)

    write_data(args.output_dir, train_pos_df=train_pos_df, train_neg_df=train_neg_df, dev_df= val_df)
    logger.info(f"Train sets done: \n\tTrain positive examples:{train_pos_df.shape}\n\tTrain negative examples:{train_neg_df.shape}")
    logger.info(f"Validation set done: {val_df.shape}")

    if args.data_splitting:
        test_df.to_json(os.path.join(args.output_dir, "test.json"), force_ascii=False, indent=4,orient='index')
        logger.info(f"Test set done: {test_df.shape}")

    logger.info("******Convert Successful******")

if __name__ == "__main__":
    main()
