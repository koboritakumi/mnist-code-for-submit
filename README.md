# mnist code for submit

これは、勉強のために書いたコードで、参考論文にあるMNISTのFC2のニューラルネットワークでの解析を実装したものです。  
FC2の線形変換をニューラルネットワークのMPO(行列積状態)に置き換えたものとなってます。

mnist.ipynb : kerasを用いて実装したもの。  
入力層(28 $\times$ 28の入力を784の一次元にしたもの) \
$\rightarrow$ 中間層1(784 $\rightarrow$ 256) \
$\rightarrow$ Relu \
$\rightarrow$ 中間層1(256 $\rightarrow$ 10) \
$\rightarrow$ Softmax \
$\rightarrow$ 出力層

そして、損失関数はcategorical crossentropyを用いた。正則化項はつけなかった。  
optimizerは、SGDを用いた。

mnist.py : ニューラルネットワーク部分をkerasを用いずに、実装したものである。

mnistSgdRevised.py : SGDをミニバッチSGDとしたものである。

mnist_mpo.py : ニューラルネットワーク部分の一部をテンソルネットワークのモデルの一つである行列積状態(MPO)に置き換えたものである。  
中間層1の行列( $784 \times 256$ )を((4,4,4),(7,4,4,4),(7,4,4,4),(4,4,4))の4つのテンソルの積に置き換える。  
中間層2の行列( $256 \times 10$ )を((4,4,1),(4,4,4,10),(4,4,4,1),(4,4,1))の4つのテンソルの積に置き換える。  

入力層(28 $\times$ 28の入力を $4\times 7\times 7\times 4$としたもの) \
$\rightarrow$ 中間層1($4\times 7\times 7\times 4$ $\rightarrow$ $4\times 4\times 4\times 4$ に変換する。) \
$\rightarrow$ Relu \
$\rightarrow$ 中間層1($4\times 4\times 4\times 4$ $\rightarrow$ $1\times 10\times 1\times 1$　に変換する。) \
$\rightarrow$ Softmax \
$\rightarrow$ 出力層

同様に、損失関数はcategorical crossentropyを用いた。正則化項はつけなかった。  
optimizerは、SGDを用いた。

しかし、このコードは時間がかかりすぎるため解析は行っていない。

mnist_mpoSgdRevised.py : mnist_mpo.pyのSGDをミニバッチSGDとしたものである。  
主にこちらの解析を行なった。

resultGraphMnistMpo.ipynb : mnist_mpoSgdRevised.pyの結果をグラフにしたものである。  
計算時間の関係で、test accuracyのみを計算した。

# 参考論文
Ze-Feng Gao et al.,"Compressing deep neural networks by matrix product operators"
Phys. Rev. Research 2, 023300 – Published 8 June 2020

