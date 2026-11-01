# 第2回 解答

## 演習

- 使ったモデル・設定（`result.json` の `config` から）：
{
  "config": {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 0.001,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {
      "do_sample": false
    }
  },
  "versions": {
    "transformers": "5.18.0",
    "peft": "0.21.1",
    "trl": "1.14.2",
    "torch": "2.14.1+cu130"
  },
  "method": "LoRA（量子化なし）",
  "before": {
    "label": "before",
    "n": 21,
    "format_rate": 0.09523809523809523,
    "content_rate": 0.0,
    "both_rate": 0.0,
    "seconds": 10.8
  },
  "after": {
    "label": "after",
    "n": 21,
    "format_rate": 1.0,
    "content_rate": 0.9523809523809523,
    "both_rate": 0.9523809523809523,
    "seconds": 22.0
  },
  "regressed": [],
  "executed_training_myself": true
}

- 自分で学習を実行したか（`executed_training_myself`）：はい

- 学習前：`format_rate` =　0.095238 / `content_rate` =　0.000000
- 学習後：`format_rate` =　1.000000 / `content_rate` =　0.952381
- 退行した質問：なし
- 学習後に残った失敗と、その型（根拠なしの質問に断定した / 根拠ありの質問に確認要求で逃げた / 形式の崩れ）：
6	PDF 以外でも出せますか、
モデルの回答："この FAQ には記載がありません。運営に確認が必要です。"
不正解理由：FAQと異なる回答。実際には、「提出形式は PDF 1 ファイル」と記載がある。
FAQに根拠がある質問に対して、その情報を回答に結びつけられず、誤って「確認が必要」と回答した。形式は正しいが、根拠の有無の判断を誤った。
- 知識を問う質問（セクション 6）の結果（学習前の分類 → 学習後の分類）と気づいたこと：
最終課題の締切	format_error → asserted
担当教員の名前	format_error → needs_review
次回の講義のテーマ	format_error → needs_review
課題の提出期限（FAQ内）	format_error → asserted
形式は改善したが、FAQに根拠があるかの判断には誤りが残った
- 学習で変わったこと・変わらなかったこと：
変わったこと：大きく回答の形式の読みやすさが向上している。
変わらなかったこと：FAQにない事実を正確に答えられるようになったわけではなく、根拠のない断定も残った。

## 宿題

- 取り組んだレベル：
基礎

- 変えた設定やデータ：
学習率を0.0002から0.001に5倍に変更した。

- 結果（比較表や数字）：

以下の各表の `seconds` は21問の評価時間を示す。

変更前： "lr": 0.0002
Training Loss：0.013761
学習時間: 50 秒 
| label  | n  | format_rate | content_rate | both_rate | seconds |
| ------ | -- | ----------- | ------------ | --------- | ------- |
| before | 21 | 0.095238    | 0.000000     | 0.000000  | 39.8    |
| after  | 21 | 1.000000    | 0.904762     | 0.904762  | 42.5    |

変更後：
"lr": 0.001
Training Loss：0.011920
学習時間：50秒
| label  | n  | format_rate | content_rate | both_rate | seconds |
| ------ | -- | ----------- | ------------ | --------- | ------- |
| before | 21 | 0.095238    | 0.000000     | 0.000000  | 10.8    |
| after  | 21 | 1.000000    | 0.952381     | 0.952381  | 22.0    |

- ファインチューニングが向く場面・向かない場面についての考察：
向く場面：利用できる情報があって、それを元にした形式を揃えた回答を生成させたいとき。
向かない場面：データから完全に間違えのない回答を生成したいとき。
