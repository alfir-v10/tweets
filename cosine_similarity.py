import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import Preprocessing
from config import params
import time
import os


def find_cosine_similarity(path='./datasets/'):
    listdir = os.listdir(path)
    if listdir:
        prep = Preprocessing()
        prep.run_preprocessing()
        sentences = prep.df.iloc[:, 0].to_numpy()
        for file in listdir:
            start_time = time.time()
            vector = pd.read_csv(os.path.join(path, file), index_col=0)
            vector.drop_duplicates(inplace=True)
            df_comp = cosine_similarity(vector.values, vector.values)
            most_common = []
            for i in range(df_comp.shape[0]):
                for j in range(i, df_comp.shape[0]):
                    if 1 > df_comp[i, j] > 0.5 and i != j:
                        most_common.append([df_comp[i, j], (i, j)])
            most_common = sorted(most_common, key=lambda x: x[0], reverse=True)[:10]

            if not os.path.exists("most_common.csv"):
                with open("most_common.csv", 'w') as f:
                    f.write("approach, sentences_one, sentences_two, cosine_similarity\n")

            with open(f"most_common.csv", 'a') as f:
                for mc in most_common:
                    sim, (i, j) = mc
                    f.write(f"{file}_pair, {sentences[i]}, {sentences[j]}, {sim}\n")
            print(file, time.time() - start_time)


if __name__ == "__main__":
    path = './datasets/'

    if os.path.exists("most_common.csv"):
        df = pd.read_csv("most_common.csv", index_col=0, header=0)
        for i in range(10):
            print(df.iloc[i, :])
    else:
        find_cosine_similarity(path)

    """
    path = './data/'
    save_sim = {}
    for config in params:
        start_time = time.time()
        prep = Preprocessing(path_data=path,
                             params=params[config])
        prep.run_preprocessing()
        vector = prep.vector
        vector.drop_duplicates(inplace=True)
        save_pairwise_sim = {}
        sentences = prep.df.iloc[:, 0].to_numpy()
        # for i in range(vector.shape[0]):
        #     for j in range(i + 1, vector.shape[0] - 1):
        #         vec_one = vector.iloc[i, :].to_numpy()
        #         vec_two = vector.iloc[j, :].to_numpy()
        #         vec_one = vec_one.reshape((1, vec_one.shape[0]))
        #         vec_two = vec_two.reshape((1, vec_two.shape[0]))
        #         cos_sim = cosine_similarity(vec_one, vec_two)
        #         if cos_sim[0][0] > 0.5:
        #             print(cos_sim, sentences[i], sentences[j])
        # save_sim_by_config.update({config: save_pairwise_sim})
        df_comp = cosine_similarity(vector.values, vector.values)
        # print(cos_sim)
        # for i in cos_sim:
        #     for j in cos_sim:
        #         save_pairwise_sim.update({str(i) + str(j): cos_sim[i][j]})
        # save_sim_by_config.update({config: save_pairwise_sim})
        # df_comp = pd.DataFrame(cos_sim, columns=sentences, index=sentences)
        most_common = []
        for i in range(df_comp.shape[0]):
            for j in range(i, df_comp.shape[0]):
                if 1 > df_comp[i, j] > 0.5 and i != j:
                    most_common.append([df_comp[i, j], (i, j)])
        most_common = sorted(most_common, key=lambda x: x[0], reverse=True)[:10]

        if not os.path.exists("most_common.csv"):
            with open("most_common.csv", 'w') as f:
                f.write("approach, sentences_one, sentences_two, cosine_similarity\n")

        with open(f"most_common.csv", 'a') as f:
            for mc in most_common:
                sim, (i, j) = mc
                f.write(f"{config}_pair, {sentences[i]}, {sentences[j]}, {sim}\n")

            # save_pairwise_sim = {}
        # if not os.path.exists("cos_sim_dataframes"):
        #     os.mkdir("cos_sim_dataframes")
        # df_comp.to_csv(f"cos_sim_dataframes/{config}.csv")
        print(config, time.time() - start_time)
    # df = pd.DataFrame(save_sim_by_config)
    # df.to_csv("cosine_sim.csv")

"""
