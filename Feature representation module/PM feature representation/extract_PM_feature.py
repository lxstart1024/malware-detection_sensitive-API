import os
import re
import torch
from UniXcoder.unixcoder import UniXcoder
import csv

def load_data(filepath):
    data_list = []
    file_list = os.listdir(filepath)
    for file in file_list:
        data_file = []
        if not os.path.isdir(file):
            with open(filepath + "/" + file, encoding="utf-8") as f:
                method_code_temp = ""
                for num, line in enumerate(f):
                    if line.startswith("="):
                        data_file.append(method_code_temp)
                        method_code_temp = ""
                    else:
                        method_code_temp = method_code_temp + re.sub(" +", " ", line).replace("\n", "").replace("None",
                                                                                                                "")
        with open(file, "a", encoding="utf-8") as file:
            for code_data in data_file:
                file.write("%s\n" % code_data)

def code_vectorizaiton(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder("microsoft/unixcoder-base")
    model.to(device)

    app_method_vec_list = []

    file_list = os.listdir(filepath)
    for file in file_list:
        if not os.path.isdir(file):
            with open(file+"_codevec.csv", mode="a", newline="") as f_csv:
                writer = csv.writer(f_csv)
                with open(filepath + "/" + file, "r", encoding="utf-8") as f:
                    for num, line in enumerate(f):
                        tokens_ids = model.tokenize([line], max_length=512, mode="<encoder-only>")
                        source_ids = torch.tensor(tokens_ids).to(device)
                        tokens_embeddings, codeSnippet_embedding = model(source_ids)
                        writer.writerow(codeSnippet_embedding.tolist())


if __name__ == '__main__':
    path = "F:/PM_data"
    load_data(path)
    code_vectorizaiton(path)