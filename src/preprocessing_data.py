import os
import re
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split

Academic_degree = "(TTƯT|PGS|TS|BS|ThS|CKII|CKI|GS)"
Middle_Name = "(St|G|J|W|R|E|L|T|N|U)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'
alphabets= "([A-Za-z])"
Upper_alphabets = "([A-Z])"

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
    # replace sentence contain \n\n latest to ""
    text = re.sub(r"\n\n(?!.*\n\n).*", "", text)
    # convert stop with "dot + space"
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)), text)

    # 
    text = text.replace("TP.", "TP<prd>")
    text = text.replace("Tp.", "Tp<prd>")
    text = text.replace("HCM.", "HCM<stop>")
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>",text)
    # text = re.sub(alphabets + "[.]" + alphabets , "\\1<prd>\\2",text)
    text = re.sub(Academic_degree+"[.]", " \\1<prd>",text)
    text = re.sub(Middle_Name + "[.]", " \\1<prd>",text)
    # text = re.sub(Upper_alphabets + "[.]", " \\1<prd>", text )
    #     
    # text = text.replace(".\n\n", "<stop> \n\n")
    text = text.replace(". ",".<stop>")
    text = text.replace("<prd>",".")

    # split sentence
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
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


# FILTER NON_EVIDENCE AND EVIDENCE
def create_evidence(df_copy):
    df = df_copy.__deepcopy__()
    df = df[df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df.reset_index(inplace=True, drop=True)
    df.loc[:,"verdict"] = "SR"
    df = df.loc[:,["id_sentence", "sentence", "claim", "verdict", "domain"]]
    return df

def create_non_evidence(df_copy):
    df = df_copy.__deepcopy__()
    df = df[~df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df.loc[:, "verdict"] = "NON"
    df.reset_index(inplace=True, drop=True)
    df = df.loc[:,["id_sentence", "sentence", "claim", "verdict", "domain"]]
    return df

def create_evidence_dev(df_copy):
    df = df_copy.__deepcopy__()
    dev_neg_df = df[~df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df.loc[:,"verdict"] = "SR"
    df.loc[df.id_sentence.isin(dev_neg_df.id_sentence),"verdict"] = "NON"
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.loc[:,["id_sentence", "id", "num_sentence", "sentence", "claim", "domain", "verdict"]]
    return df


# FILTER SUPPORTED AND REFUTED

def create_sr(df_copy):
    df = df_copy.__deepcopy__()
    df = df[df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df.reset_index(inplace=True, drop=True)
    df = df.loc[:,["id_sentence", "sentence", "claim", "verdict", "domain"]]
    return df

def create_sup(sr_df):
    df = sr_df[sr_df.verdict == "SUPPORTED"]
    df.reset_index(inplace=True, drop=True)
    df = df.loc[:,["id_sentence", "sentence", "claim", "verdict", "domain"]]
    return df

def create_refuted(sr_df):
    df = sr_df[sr_df.verdict == "REFUTED"]
    df.reset_index(inplace=True, drop=True)
    df = df.loc[:,["id_sentence", "sentence", "claim", "verdict", "domain"]]
    return df

def create_sentence_sr_dev(val_df):
    df = val_df[val_df.apply(lambda row: evidence_isIn_sentence(row), axis=1)]
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.loc[:,["id_sentence", "id", "num_sentence", "sentence", "claim", "domain", "verdict"]]
    return df


def write_data_classifier_evidence(output_dir, train_evidence_df, train_non_evidence_df, dev_evidence_df):
    train_evidence_df.to_json(os.path.join(output_dir, "train_evidence.json"), force_ascii=False, indent=4, orient='index')
    train_non_evidence_df.to_json(os.path.join(output_dir, "train_non_evidence.json"), force_ascii=False, indent=4,orient='index')
    dev_evidence_df.to_json(os.path.join(output_dir, "dev_classifier_evidence.json"), force_ascii=False, indent=4,orient='index')


def write_data_claim_verification(output_dir, train_sup_df, train_refured_df, dev_sr_df):
    train_sup_df.to_json(os.path.join(output_dir, "train_sup_sentences.json"), force_ascii=False, indent=4, orient='index')
    train_refured_df.to_json(os.path.join(output_dir, "train_refuted_sentences.json"), force_ascii=False, indent=4,orient='index')
    dev_sr_df.to_json(os.path.join(output_dir, "dev_sentences.json"), force_ascii=False, indent=4,orient='index')


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
                    default=True,
                    # action='store_true',
                    help="Data splitting. True: train set, test set and validation set. False: train set, validation set")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
    # Raise a warning or display a message
        print("Warning: The specified data directory does not exist.")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info("=============== EVIDENCE DATA ===============")
    df = read_data(args.data_dir)

    train_df, val_df = train_test_split(df, train_size=0.7, random_state=42)
    if args.data_splitting:
        val_df, test_df = train_test_split(val_df, train_size=0.7, random_state=42)
    train_df = process_data(train_df)
    val_df = process_data(val_df)
    
    # CREATE DATASETS EVIDENCE AND NON-EVIDENCE
    train_evidence_df = create_evidence(train_df)
    train_non_evidence_df = create_non_evidence(train_df)

    dev_evidence_df = create_evidence_dev(val_df)

    write_data_classifier_evidence(args.output_dir, train_evidence_df=train_evidence_df, train_non_evidence_df=train_non_evidence_df, dev_evidence_df= dev_evidence_df)

    logger.info(f"Train classifier evidence sets done: \n\tTrain evidence examples:{train_evidence_df.shape}\n\tTrain non-evidence examples:{train_non_evidence_df.shape}")
    logger.info(f"Validation set done: {dev_evidence_df.shape}")
    
    logger.info("=============== SUPPORTED AND REFUTED DATA ===============")
    # CREATE DATASETS SUPPORTED AND REFUTED

    sr_df = create_sr(train_df)
    train_sup_df = create_sup(sr_df)
    train_ref_df = create_refuted(sr_df)
    dev_sr_df = create_sentence_sr_dev(val_df)

    write_data_claim_verification(args.output_dir, train_sup_df=train_sup_df, train_refured_df=train_ref_df, dev_sr_df= dev_sr_df)

    logger.info(f"Train supported-refuted sets done: \n\tTrain supported examples:{train_sup_df.shape}\n\tTrain refuted examples:{train_ref_df.shape}")
    logger.info(f"Validation set done: {dev_sr_df.shape}")

    if args.data_splitting:
        test_df.to_json(os.path.join(args.output_dir, "test.json"), force_ascii=False, indent=4,orient='index')
        logger.info(f"Test set done: {test_df.shape}")

    logger.info("******Convert Successful******")

if __name__ == "__main__":
    main()

