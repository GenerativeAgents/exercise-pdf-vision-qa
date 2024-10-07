# GPT-4o の Vision を使用した PDF への Q&A

## このリポジトリのコード

[app/main.py](app/main.py) に、PDF への Q&A アプリケーションのサンプルコードがあります。

このアプリケーションでは、PDF ファイルから抽出したテキストを検索したうえで、LLM (GPT-4o) に質問できます。

PDF ファイルから抽出したテキストは、ページ単位でチャンク化され、OpenAI の Embeddings API を使ったベクトルの類似度で検索されます。

## ワーク

PDF ファイルの内容をテキスト化する処理は、なかなかうまくいかないことも多いです。

たとえば、テキスト化した際に表などの構造が崩れてしまうことで、LLM が適切に回答できないことがあります。

そこで、GPT-4o の Vision を使用して、PDF ファイルの内容を画像として認識することで、より正確な応答を得られないか検証したいです。

[app/main.py](app/main.py) をベースとして、GPT-4o の Vision を使用した画像認識も踏まえて回答するようにしてください。

## セットアップ

1. Visual Studio Code で Dev Container を起動してください。
2. [.env.template](.env.template) ファイルをコピーして、.env ファイルを作成し、OpenAI の API キーを設定してください。
3. 適当な PDF ファイルをダウンロードしてください。

## 実行方法

以下のコマンドで実行してください。

```console
poetry run python app/main.py --file <PDFファイルのパス>
```

すると、指定した PDF の各ページのテキストが順にインデクシングされ、その後に質問を受け付るようになります。

```
2024-10-07 05:26:43,302 - __main__ - INFO - Indexing PDF <PDFファイルのパス>
2024-10-07 05:26:43,423 - __main__ - INFO - Indexing page 1 / 100
    :
2024-10-07 05:26:47,846 - __main__ - INFO - Indexing completed
Question:
```

> [!WARNING]
> ワークの際に実行時間がかかりすぎないよう、PDF ファイルは先頭 10 ページのみを検索対象としてインデクシングするようになっています。

質問 (Question) を入力すると、以下のように検索結果と LLM の回答が表示されます。

```
### Search result 1 ###
    :
### Search result 2 ###
    :
### Search result 3 ###
    :
Answer: ...
```
