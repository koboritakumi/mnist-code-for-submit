# mnist code for submit

これは、勉強のために書いたコードです。
参考論文にあるMNISTのFC2のニューラルネットワークでの解析を実装したものです。
FC2の線形変換をニューラルネットワークのMPO(行列積状態)に置き換えたとなってます。

mnist.ipynb : kerasを用いて実装したものです。
入力層(28$\times$28の入力を784の一次元にしたもの)$\rightarrow$中間層1(784$\rightarrow$256)$\rightarrow$

mnist.py : ニューラルネットワーク部分