## 元のレポジトリ

https://github.com/podgorskiy/ALAE

2020/12/07/22:10 JSTのもの

## 使い方

### 初期設定
1. ALAE直下へ移動
2. export PYTHONPATH=$PYTHONPATH:$(pwd)
3. このレポジトリにあるmake_generation_figure_own.pyとmake_recon_figure_STL10.pyをmake_figuresフォルダへ移動
4. prepare_STL10_tfrecords_tf1v2.pyをdataset_preparationフォルダへ移動
5. STL10.yamlはconfigフォルダへ移動

### データセットの用意方法

以下STL10の場合

1. このレポジトリにあるTFフォルダ内のDockerfileから環境作成
2. configフォルダなどと同じ階層にDatasetフォルダを作成
3. Datasetフォルダ直下にSTL10フォルダとSTL10_testフォルダを用意
4. STL10はラベル無しデータ10万, STL10_testはラベルデータ800*10(それぞれフォルダ分けされている)
5. configフォルダ内にSTL10.yamlを用意(中のパスなどがこの後使われる)
6. dataset_preparation内のprepare_STL10_tfrecords_tf1v2.pyを実行する
7. configフォルダなどと同じ階層にdataフォルダが作られる(data/datasets/STL10/tfrecordsなど)

以下デフォルトではconfigファイルのモデルが用いられる
このレポジトリにあるDokcerfile(データセット用意で用いたものと別)から環境作成

### 再構築

1. dataset_samples/STL10フォルダを作成, 直下に再構築させる画像を配置 プログラム的には41枚想定(106, 138行目
2. python make_figures/make_recon_figure_STL10.pyを実行
3. make_figuresの下に元画像,再構築画像*5が並んだ1枚の画像ファイルが作られる

### 生成

1. python make_figures/make_generation_figure_own.pyを実行
2. make_figuresの下に生成画像24枚が並んだ1枚の画像ファイルが作られる

### 学習

※学習の段階で解像度を上げていくので学習当初は動いていても途中でメモリーが足りなくなることがあるので注意. 学習開始時でメモリギリギリだとたぶん途中で止まる.

1. dataset_samples/STL10フォルダを作成, 配下にSTL10の画像を数十枚配置 128\*128のサイズでないとエラーになるので注意(256\* 256でも可?) STL10のデフォルトが96\*96なので適当にリサイズする
2. python train_alae.py -c STL10
3. training_artifactsの下に成果物が出てくる
