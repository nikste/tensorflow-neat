"""builds graph from genes"""
import tensorflow as tf
import numpy as np
import sys

from Constants import OUTPUT0, OUTPUT1, INPUT0, INPUT1
from data_fetcher import generate_xor


def add_node(inputs,name="stdname"):
    # print "adding node, for length of input:",len(inputs)
    # init_vals = tf.truncated_normal([len(inputs), 1], stddev=1. / math.sqrt(2))
    with tf.name_scope(name):
        init_vals = tf.truncated_normal([len(inputs), 1], stddev=1. / math.sqrt(2))
        w = tf.Variable(init_vals)
        b = tf.Variable(tf.zeros([1]))

        if len(inputs) > 1:
            in_tensor = tf.transpose(tf.squeeze(tf.pack(inputs)))
            output = tf.nn.relu(tf.matmul(in_tensor, w) + b, name=name)
            return output

        else:
            # in_tensor = tf.squeeze(tf.pack(inputs))
            in_tensor = tf.squeeze(tf.pack(inputs))
            output = tf.transpose(tf.nn.relu(tf.mul(in_tensor, w) + b, name=name))
            return output


def build_and_test(connections, genotype, x, y, x_test, y_test, run_id="1"):

    with tf.name_scope("input"):
        x0 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x0")
        x1 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x1")

    with tf.name_scope("ground_truth"):
        y_ = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="y_")

    # connections contains only (innovation_num, from, to)
    # genotype contains only {innovation_num: True/False)

    # need a list containing [node, inputs]
    # for all the same to's from connections collect from's (where genotype exists and says enabled)
    # connection can only exist from lower number to higher in the network


    # filter out disabled and non existent genes in this phenotype
    genotype_keys = sorted(genotype.keys())

    # filter connections
    exisiting_connections = []
    for i in xrange(0,len(genotype_keys)):
        if genotype[genotype_keys[i]]:
            exisiting_connections.append(connections[genotype_keys[i]])

    # collect nodes and connections: sort by to field from connections
    connections_sorted = sorted(exisiting_connections, key=lambda connections: connections[2])

    # merge the same nodes
    connections_merged = [[connections_sorted[0][2],[connections_sorted[0][1]]]]
    for i in xrange(1,len(connections_sorted)):
        # same as last node
        if connections_sorted[i][2] == connections_merged[-1][0]:
            connections_merged[-1][1].append(connections_sorted[i][1])
        else:
            connections_merged.append([connections_sorted[i][2],[connections_sorted[i][1]]])

    tf_nodes_dict = {INPUT0: x0, INPUT1: x1}
    for cn in connections_merged:
        node_cons = cn[1]
        node_id = cn[0]
        tf_input_nodes = [tf_nodes_dict[node_key] for node_key in node_cons]
        node_name = str(node_id) + "_"
        for na in node_cons:
            node_name += "_" + str(na)
        tf_nodes_dict[cn[0]] = add_node(tf_input_nodes, name=node_name)

    num_nodes = tf.constant(len(tf_nodes_dict.keys()))
    tf.scalar_summary("num_nodes", num_nodes)

    with tf.name_scope("softmax_output"):
        # requery output nodes to add to optimization
        output_0 = tf_nodes_dict[OUTPUT0]
        output_1 = tf_nodes_dict[OUTPUT1]

        output_final_pre = tf.transpose(tf.squeeze(tf.pack([output_0,output_1])))

        W_o1 = tf.Variable(tf.truncated_normal([2, 2], stddev=1. / math.sqrt(2)))
        b_o1 = tf.Variable(tf.zeros([2]))
        output_final = tf.nn.softmax(tf.matmul(output_final_pre,W_o1) + b_o1, name="output_softmax")

    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(output_final, y_)
        loss = tf.reduce_mean(cross_entropy)
        tf.scalar_summary("loss",loss)

    with tf.name_scope("optimizer"):
        opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(output_final, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy = tf.reduce_mean(tf.abs(output_final - y_))
        tf.scalar_summary("accuracy", accuracy)

    init = tf.initialize_all_variables()
    sess = tf.Session()

    tf.merge_all_summaries()
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./train/' + run_id,
                                          sess.graph)

    sess.run(init)

    for i in xrange(1000):
        _, loss_val, summary = sess.run([opt, loss, merged], feed_dict={x0: np.expand_dims(x[:, 0], 1),x1: np.expand_dims(x[:, 1], 1), y_: y})
        if i % 100 == 0:
            train_writer.add_summary(summary, i)

    acc_test = sess.run([accuracy], feed_dict={x0: np.expand_dims(x_test[:, 0], 1), x1: np.expand_dims(x_test[:, 1], 1), y_: y_test})
    sess.close()
    tf.reset_default_graph()
    return acc_test[0]




def build_and_test_back(Connections, x, y, x_test, y_test):
    x0 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x0")
    x1 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x1")

    y_ = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="y_")

    output_l1_0, w_l1_0, _ = add_node([x0, x1], name="output_l1_0")
    output_l1_1, w_l1_1, _ = add_node([x0, x1], name="output_l1_1")


    output_final_pre = tf.transpose(tf.squeeze(tf.pack([output_l1_0,output_l1_1])))


    W_o1 = tf.Variable(tf.truncated_normal([2, 2], stddev=1. / math.sqrt(2)),dtype=tf.float32)
    b_o1 = tf.Variable(tf.zeros([2]))
    output_final = tf.nn.softmax(tf.matmul(output_final_pre,W_o1) + b_o1, name="output_softmax")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(output_final, y_)
    loss = tf.reduce_mean(cross_entropy)



    opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


    accuracy = tf.reduce_mean(tf.abs(output_final - y_))

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]],dtype=np.float32)
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]],dtype=np.float32)

    x_test = x.copy()
    y_test = y.copy()
    for i in xrange(10000):
        r = np.random.permutation(len(x))
        x = x[r]
        y = y[r]
        _, loss_val = sess.run([opt, loss], feed_dict={x0: np.expand_dims(x[:, 0], 1),
                                                       x1: np.expand_dims(x[:, 1], 1), y_: y})
        # print i, loss_val
        if i % 1000 == 0:
            print "Step:", i, "Current loss:", loss_val
            ou = sess.run([output_final], feed_dict={x0: np.expand_dims(x_test[:, 0], 1),
                                                     x1: np.expand_dims(x_test[:, 1], 1), y_: y_test})

            ou = ou[0]

            for inin, yy, ouou in zip(x_test,y_test,ou):
                print inin, yy, ouou



    # batch_size = 100
    # print x.shape
    # print x_test.shape
    # for i in range(100000):
    #
    #     for batch_start in range(0,x.shape[0],batch_size):
    #         batch_end = batch_start + batch_size
    #         in0 = x[batch_start: batch_end, 0]
    #         in1 = x[batch_start: batch_end, 1]
    #         out0 = y[batch_start:batch_end]
    #
    #         acc, out, l, _ = sess.run([accuracy, output_final, loss, opt], feed_dict={x0: in0, x1: in1, y_: out0})
    #         # for oo,gg in zip(out0,out):
    #         #     print(oo, gg[0])
    #         # print acc
    #     if i% 10 == 0:
    #         acc_train = sess.run([accuracy], feed_dict={x0:x[:,0], x1:x[:,1], y_: y})
    #
    #         acc_test, out = sess.run([accuracy, output_final], feed_dict={x0: x_test[:, 0], x1: x_test[:, 1], y_: y_test})
    #         print "iteration:",i,acc_train, acc_test
    #
    #
    #
    #         for xs,oo,gg in zip(x_test, out0,out):
    #             print xs,oo, gg[0]
    #         print ""
    #         # with sess.as_default():
    #         #     print "test error:", accuracy.eval(feed_dict={x0: x_test[:, 0], x1: x_test[:, 1], y_: y_test})
    #
    #         # print i,"training error:",acc,l
    #         # print in0[0],in1[0],out0[0],out[0]
    #         # print in0[1],in1[1],out0[1],out[1]
    #         # li = [c for c in zip(out,y)]
    #         # for lii in li:
    #         #     print lii[0],lii[1]
    #     # print w_
    #     # print np.ravel(out)
    #     # print y
    # #


import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))



def test_build_graph():
    x, y = generate_xor(n_samples=1000)
    x_test, y_test = generate_xor(n_samples=10)

    print x.shape
    print y.shape
    x0 = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="x0")
    x1 = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="x1")

    y_ = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="y_")



    # requery output nodes to add to optimization


    intermediate = add_node([x0])
    output_0 = add_node([x0,x1])
    output_1 = add_node([x0,intermediate])

    output_final_pre = tf.transpose(tf.squeeze(tf.pack([output_0, output_1])))

    W_o1 = tf.Variable(tf.truncated_normal([2, 2], stddev=1. / math.sqrt(2)))
    b_o1 = tf.Variable(tf.zeros([2]))
    output_final = tf.nn.softmax(tf.matmul(output_final_pre, W_o1) + b_o1, name="output_softmax")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(output_final, y_)
    loss = tf.reduce_mean(cross_entropy)

    opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    accuracy = tf.reduce_mean(tf.abs(output_final - y_))

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in xrange(1000):
        r = np.random.permutation(len(x))
        x = x[r]
        y = y[r]
        _, loss_val = sess.run([opt, loss], feed_dict={x0: np.expand_dims(x[:, 0], 1),
                                                       x1: np.expand_dims(x[:, 1], 1), y_: y})
        print loss_val
        # print i, loss_val
        # if i % 1000 == 0:
        #     print "Step:", i, "Current loss:", loss_val
        #     ou = sess.run([output_final], feed_dict={x0: np.expand_dims(x_test[:, 0], 1),
        #                                              x1: np.expand_dims(x_test[:, 1], 1), y_: y_test})
        #
        #     ou = ou[0]
        #
        #     for inin, yy, ouou in zip(x_test, y_test, ou):
        #         print inin, yy, ouou

    acc = sess.run([accuracy], feed_dict={x0: np.expand_dims(x_test[:, 0], 1),
                                          x1: np.expand_dims(x_test[:, 1], 1), y_: y_test})
if __name__ == "__main__":
    pass
    # test_build_graph()
    # X, y = generate_xor(n_samples=1000)
    # X_test, y_test = generate_xor(n_samples=10)



    # print "train"
    # for xs,y in zip(X,y):
    #     print xs,y
    #
    # print "test"
    # for xs,y in zip(X_test,y_test):
    #     print xs,y

    # build_and_test(X, y, X_test, y_test)