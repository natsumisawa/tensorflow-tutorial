import tensorflow as tf
sess = tf.InteractiveSession()

log_dir = '/tmp/tensorflow/mnist/logs/simple01'

# 指定したディレクトリがあれば削除し、再作成
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

# 定数で1 + 2
x = tf.constant(1, name='x')
y = tf.constant(2, name='y')
z = x + y

# このコマンドでzをグラフ上に出力
_ = tf.summary.scalar('z', z)

# SummaryWriterでグラフを書く(これより後のコマンドはグラフに出力されない)
summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

# 実行
print(sess.run(z))

# SummaryWriterクローズ
summary_writer.close()
# TensorBoard情報出力ディレクトリ
