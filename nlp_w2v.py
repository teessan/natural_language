# /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from janome.tokenizer import Tokenizer
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

t = Tokenizer()

def make_word_corpus(line):
    '''
    引数：line -> 分かち書きするための文字列

    取得した文字列（line）を分かち書きして返す処理
    名詞、一般がない場合は、空白を返す(' ')
    '''
    words = ""
    tmp = []
    for ix,token in enumerate(t.tokenize(line)):
        index = token.part_of_speech.find("名詞,一般")
        if index != -1:
            words += token.base_form
            words += " "
    tmp=words.split(" ")

    return tmp

def make_model_w2v_data(data,model_name):
    '''
    引数：data -> data list
    引数：model_name  -> モデルファイル名

    word2vecのmodelファイル作成用関数（メモリに保有したデータから作成）
    '''
    if(len(data)==0):
        print("No charactor for model ")
        sys.exit()
    else:
        model = word2vec.Word2Vec(data, size=100, window=5, hs=1, min_count=1, sg=1)
        model.save(model_name)



def search_words(word,top_num,model):
    '''
    引数：word -> 類似単語を出力したい単語
    引数：top_num -> 出力したい類似単語の個数指定
    引数：model -> ロードされたモデルオブジェクト

    引数のワードとコサイン類似度が高い順にtop_num個出力する関数
    '''
    result = model.most_similar(positive=word, topn=top_num)
    return result
    #n=[cos_word[0] for cos_word in result]

def load_model(model_name):
    '''
    引数：model_name -> モデルファイル名

    modelをロードした、モデルオブジェクトを返す関数
    '''
    model = word2vec.Word2Vec.load(model_name)
    return model


if __name__ == '__main__':
    data = []
    model_name = "w2v.model"
    vocab_thresh = 0   # 出力する単語を

    with open("Cinderella.txt","r",encoding="utf-8") as f:
        data = f.readlines()

    for idx,i in enumerate(data):
        data[idx] = i.replace("\n","")

    corpus = [make_word_corpus(i) for i in data]
    print(corpus)

    make_model_w2v_data(corpus,model_name)
    model = load_model(model_name)
    #result = search_words("姉さん",10,model)
    #print(result)


    #ボキャブラリー一覧
    vocabs = []
    for word , vocab_obj in model.wv.vocab.items():
        if vocab_obj.count >= vocab_thresh:    # N回以上登場する閾値
            vocabs.append( word )
        print(word)

    n_vocab = len(vocabs)
    print("単語数={}".format(n_vocab))

    emb_tuple = tuple([model[v] for v in vocabs])
    X = np.vstack(emb_tuple)
    models = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    models.fit_transform(X)

    # matplotlibで t-SNEの図を描く
    skip = 0
    limit = n_vocab  # 全単語出力

    # 日本語フォント対応
    #書く環境で変更する必要あり
    fp = FontProperties(fname='/System/Library/Fonts/ヒラギノ角ゴシック W0.ttc', size=14)
    #fp = FontProperties(fname='C:\Windows\Fonts\HGRGM.TTC', size=14)

    plt.figure( figsize=(50,30) )
    plt.scatter( models.embedding_[skip:limit, 0], models.embedding_[skip:limit, 1] )

    count = 0
    for label, x, y in zip(vocabs, models.embedding_[:, 0], models.embedding_[:, 1]):
        count +=1
        if( count < skip ): continue
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',fontproperties=fp)
        if( count == limit ): break

    plt.savefig('test.png')
