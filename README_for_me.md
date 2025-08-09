# Information
### functional_from_freq_IR_Raman.py
- 3つを縦につなげて、conv層を使って3→768に変換後、encoder_transforemrを適用 (実質線形変換だが、[batch_size, 3, max_length]→[batch_size, 768, max_length]の変換なので、conv層を用いている)

### Error Handling Note
- conv1dの出力が学習途中でnanになる→lr 1e-5, clip grad norm 1.0に変更

### ルール
- パスは絶対パスで

### module
- SmilesTrainerFreqIrRamanFinetune: batch_balidationのlossはteacher foringで計算したもの(trainと合わせるため) acc_tensor(reconstruction_rate_tensor)はgreedyで計算したもの(reconstruction rateはgreedyで計算しないと意味ないため)

# reconstructionの判定について
- smiles文字列に変換してから判定を行う。idで判定するおと1.attention maskがFALSEの部分の扱いがめんどくさい。 2. token２つがつながって一つのtokenになるパターンを考慮できていないという理由から。
- Convert the ids to smiles string before evaluating these two are eqivalent or not. Because 1. the handling of the part where attention mask is false is a pain in the ass. 2. it is not consderable the patern where connected 2 tokens become single anothe token.

