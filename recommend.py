# -*- coding: utf-8 -*-
#from typing import Dict, Text
import sys

import tensorflow_recommenders as tfrs

import tensorflow as tf
import numpy as np

import pandas as pd
import pymysql




def recommend_recipes(user_id):

    connection = pymysql.connect(
        user="megu",
        password="0000",
        host="localhost",
        port=3306,
        database="board"
    )

    cursor = connection.cursor()

    # 이 부분에서 SQL 쿼리를 실행
    cursor.execute("""
                SELECT * FROM tbl_board WHERE status = 'Y';
             """)

    result1 = cursor.fetchall()
    user_data1 = pd.DataFrame(result1)

    # 나머지 코드 실행
    recipes = pd.DataFrame(result1)
    #columns로 몇번째 열에 어떤 컬럼이 있는지 명시해주기
    recipes.rename(columns={0: 'bno'}, inplace=True)
    recipes.rename(columns={1: 'title'}, inplace=True)
    #print(recipes)


    cursor.execute("""
                SELECT * FROM tbl_post;
                """)

    result1 = cursor.fetchall()
    user_data2 = pd.DataFrame(result1)
    post = pd.DataFrame(result1)
    post.rename(columns = {1 : 'bno'}, inplace = True)
    post.rename(columns = {7 : 'userId'}, inplace = True)
    #print(post)

    # 'tbl_post.csv'에서 'bno'와 'userid' 열을 선택
    # post = pd.read_csv('lesson_data')
    post['userId'] = post['userId'].astype(str)
    post = post[['bno', 'userId']]

    # 'tbl_board.csv'에서 'bno'와 'title' 열을 선택
    # board_df = pd.read_csv('tbl_board.csv')
    recipes['title'] = recipes['title'].astype(str)
    recipes = recipes[['bno', 'title']]

    # 'bno'를 기준으로 두 데이터 프레임을 병합
    merged_data = post.merge(recipes, on='bno', how='inner')
    #print(merged_data)

    # # 필요한 열 선택
    post = merged_data[['title', 'userId']]
    recipes = recipes[['bno', 'title']]

    def pandas_to_dataset(dataframe, target_column_name):
        dataframe = dataframe.copy()
        labels = dataframe.pop(target_column_name)  # 타겟 열 선택
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        return dataset
    ratings_dataset = pandas_to_dataset(post,'userId')
    recipes_dataset = pandas_to_dataset(recipes, 'title')

    # Create a function to convert a row to the desired format
    def map_features(x,y):
        return {
            "title": x["title"],
            "userId": y
        }

    # # Apply the mapping function to each row in the dataset
    ratings_dataset = ratings_dataset.map(map_features)
    recipes_dataset = tf.data.Dataset.from_tensor_slices(recipes['title'])

    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings_dataset.map(lambda x : x['userId']))

    recipes_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    recipes_titles_vocabulary.adapt(recipes_dataset)

    class MovieLensModel(tfrs.Model):
        def __init__(self, user_model, recipes_model, task):
            super(MovieLensModel, self).__init__()


            # Set up user and movie representations.
            self.user_model = user_model
            self.recipes_model = recipes_model

            # Set up a retrieval task.
            self.task = task

        def compute_loss(self, features, training=False):


            # Define how the loss is computed.

            user_embeddings = self.user_model(features["userId"])
            recipes_embeddings = self.recipes_model(features["title"])

            return self.task(user_embeddings, recipes_embeddings)

    user_model = tf.keras.Sequential([
        user_ids_vocabulary,
        tf.keras.layers.Embedding(input_dim=user_ids_vocabulary.vocab_size(), output_dim=64)
    ])

    recipes_model = tf.keras.Sequential([
        recipes_titles_vocabulary,
        tf.keras.layers.Embedding(input_dim=recipes_titles_vocabulary.vocab_size(), output_dim=64)
    ])
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        recipes_dataset.batch(128).map(recipes_model)
    ))

    model = MovieLensModel(user_model, recipes_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    # Train for 3 epochs.
    model.fit(ratings_dataset.batch(4096), epochs=5, verbose=0)
    userId = user_id


    # 기존에 작성한 게시글의 제목을 가져옵니다.
    user_titles = merged_data[merged_data['userId'] == userId]['title'].unique()

    # TensorFlow 데이터셋에서 사용자가 이미 작성한 제목을 필터링합니다.
    recipes_dataset = recipes_dataset.filter(
        lambda title: tf.math.logical_not(tf.math.reduce_any(tf.math.equal(title, user_titles)))
    )

    # BruteForce 인덱스를 생성하고 데이터셋으로부터 인덱싱합니다.
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        recipes_dataset.batch(100).map(lambda title: (title, model.recipes_model(title)))
    )

    # print("Debug: Something important happened here")
    # 사용자에게 추천합니다. 중복을 피하기 위해 루프를 사용합니다.
    recommended_titles = set()
    user_id = np.array([userId], dtype=object)
    i = 0
    while len(recommended_titles) < 5:  # 4개의 유니크한 추천을 원합니다.
        _, titles = index(user_id)
        for title in titles[0, i:i+1]:  # 다음 추천을 가져옵니다.
            title_text = title.numpy().decode('utf-8')
            if title_text not in recommended_titles:  # 추천이 유니크한지 확인합니다.
                recommended_titles.add(title_text)
            if len(recommended_titles) == 5:  # 원하는 수의 추천을 얻었다면 루프를 종료합니다.
                break
        i += 1  # 다음 추천을 위해 인덱스를 증가시킵니다.

    bno_dict = dict(zip(recipes['title'], recipes['bno']))

    bno_list = [bno_dict.get(title, None) for title in recommended_titles]

    # 'bno' 리스트 출력
    for bno in bno_list:
        if bno is not None:
            print(bno)

#  # 추천 결과를 리스트로 변환합니다.
#  titles = list(recommended_titles)
#
#  #titles = [title.numpy().decode('utf-8') for title in titles[0, :4]]
#
#  # titles에 해당하는 bno 찾기
#  bno_list = []
#
#  for title in titles:
#      # recipes 데이터프레임에서 해당 타이틀에 해당하는 bno를 찾습니다.
#      bno = recipes[recipes['title'] == title]['bno'].values
#      if len(bno) > 0:
#          bno_list.append(bno[0])
#      else:
#          bno_list.append(None)
#
#  # 결과 출력
#  #recommendations = pd.DataFrame({'userId':userId, 'title': titles, 'bno': bno_list})
# # print("Debug: Something important happened here")
#  recommendations = pd.DataFrame(bno_list)
#
#  # DataFrame recommendations를 문자열로 변환하고, 각 행을 쉼표로 구분하여 출력
#
#  connection.close()
#  result = recommendations.to_csv(index=False, header=False)
#  print(result)
#print(recommendations.to_csv(index=False, header=False))
# sys.stdout.flush()
# return recommendations.to_csv(index=False, header=False)

if __name__ == "__main__":
    # 명령줄 인수에서 userId 값을 읽어옴
    if len(sys.argv) < 2:
        print("Usage: python recommend.py <userId>")
        sys.exit(1)

    user_id = sys.argv[1]
    recommend_recipes(user_id)

