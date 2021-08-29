import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential


def preprocess(x, y):
    """数据处理函数"""
    #转换数据类型 x范围为0~255 将其变为0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


# 加载数据
(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)

# 处理train数据
batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
#将数据顺序打乱 并抽取batch_size大小数据进行训练
db = db.map(preprocess).shuffle(10000).batch(batch_size)

# 处理test数据
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)

# 生成train数据的迭代器
db_iter = iter(db)
sample = next(db_iter)
print(f'batch: {sample[0].shape,sample[1].shape}')

# 设计网络结构
# 五层全连接层(其中四层的激活函数为relu)
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b,784] --> [b,256]
    layers.Dense(128, activation=tf.nn.relu),  # [b,256] --> [b,128]
    layers.Dense(64, activation=tf.nn.relu),  # [b,128] --> [b,64]
    layers.Dense(32, activation=tf.nn.relu),  # [b,64] --> [b,32]
    layers.Dense(10),  # [b,32] --> [b,10], 330=32*10+10
])

model.build(input_shape=[None, 28 * 28])
model.summary()  # 调试
# w = w - lr*grad
optimizer = optimizers.Adam(lr=1e-3)  # 优化器，加快训练速度 lr为超参数


#进行训练
for epoch in range(10):

    for step, (x, y) in enumerate(db):

        # x:[b,28,28] --> [b,784]
        # y:[b]
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            # [b,784] --> [b,10]
            logits = model(x)
            y_onehot = tf.one_hot(y, depth=10)
            # [b]
            loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
            loss_ce = tf.reduce_mean(
                tf.losses.categorical_crossentropy(y_onehot,
                                                   logits,
                                                   from_logits=True))

        grads = tape.gradient(loss_ce, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, f'loss: {float(loss_ce),float(loss_mse)}')

    # 进行测试并运算精确值
    total_correct = 0
    total_num = 0
    for x, y in db_test:
        # x:[b,28,28] --> [b,784]
        # y:[b]
        x = tf.reshape(x, [-1, 28 * 28])
        # [b,10]
        logits = model(x)
        # logits --> prob [b,10]
        prob = tf.nn.softmax(logits, axis=1)
        # [b,10] --> [b], int32
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        # pred:[b]
        # y:[b]
        # correct: [b], True: equal; False: not equal
        correct = tf.equal(pred, y)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

        total_correct += int(correct)
        total_num += x.shape[0]

    acc = total_correct / total_num
    print(epoch, f'test acc: {acc}')



