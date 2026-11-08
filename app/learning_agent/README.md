# 学習支援エージェント

OpenAI Responses API と `gpt-6-luna` を使用します。既存の `.env` などにある `OPENAI_API_KEY` を利用します。

リポジトリのルートで次を実行すると、対話中の待機時間やツールの実行状況が表示されます。

```bash
uv run python3 -m app.learning_agent
```

`exit` または `Ctrl+D` で終了し、応答待ちの途中は `Ctrl+C` で中断できます。1回だけ質問することもできます。

```bash
uv run python3 -m app.learning_agent "第2回の用語から確認問題を1問出して"
```

ADK の Web 画面で使う場合は、リポジトリのルートで次を実行します。

```bash
uv run adk web app/learning_agent
```

5秒ごとの「待機中」は応答を受け取っていないことを示し、通信やモデルが正常に進行していることを保証するものではありません。質問1回の待機上限は60秒です。変更する場合は、例えば `uv run python3 -m app.learning_agent --timeout 120` とします。この表示は上記のコマンド用で、ADK の Web 画面には追加されません。
