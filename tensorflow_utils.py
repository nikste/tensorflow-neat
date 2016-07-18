"""builds graph from genes"""
import tensorflow as tf
import numpy as np
import math
from Constants import OUTPUT0, OUTPUT1, INPUT0, INPUT1

def add_node(inputs,name="stdname"):
    with tf.name_scope(name):
        init_vals = tf.truncated_normal([len(inputs), 1], stddev=1. / math.sqrt(2))
        w = tf.Variable(init_vals)
        b = tf.Variable(tf.zeros([1]))

        if len(inputs) > 1:
            in_tensor = tf.transpose(tf.squeeze(tf.pack(inputs)))
            output = tf.nn.relu(tf.matmul(in_tensor, w) + b, name=name)
            return output

        else:
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