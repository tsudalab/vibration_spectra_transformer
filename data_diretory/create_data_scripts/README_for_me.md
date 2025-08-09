# information

- このディレクトリではattention_maskは無視しないものを1, 無視するもの(マスクされるもの)を0とする。(smilesのattention_maskでも、dataのattention_maskでも)pytorchのtransformer_encoderにinputする際には無視するものをTrue, 無視しないものをFalseにする必要がある。(要確認)
- 先頭にCLSを入れてなかったので、poolingはattentionが1の部分の平均　(今は先頭になっているので変えなきゃ)
- freq_attention_masksという名前だが、freq, IR, Ramanは同じ構造なので、IR, Ramanにも使える
- heavy atom 26以上ではtrain/valid/testの指定もディレクトリ分けも行わない。今は評価にしか使わないから

### それぞれのデータ数

| heavy atom数 | データサイズ |
| ---- | ---- |
| 26 | 4312 (225以下のもの) |
| 36 | 768 (226以上は84個) |

### smiles length
※ (lengthの検証に使われる) tokenizer.tokenizeはCLS, EOSを含まない
※ (data作成に使われる) call はEOS, CLSを自動で加える

| num of heavy atoms | max |
| ---- | ---- |
| 5-25 | 26 (CLS, EOS含まず)|
|モデルに使ってる値| (32CLS, EOS含んで)|

- GDB9のmax_spectrum_lengthは100に設定。最大値自体は81だったが、外挿も考えて
- MMP05percentのheavy_atom12ではじめて100を超えたので、そんなもん