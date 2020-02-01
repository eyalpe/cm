import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


# frozen_graph = freeze_session(K.get_session(),
#                               output_names=[out.op.name for out in model.outputs])

if __name__=="__main__":
    import tensorflow as tf
    from tensorflow.python.platform import gfile

    f = gfile.FastGFile("./model/tf_model.pb", 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()
    sess = tf.Session()
    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)
    #all operations
    sess.graph.get_operations()
    softmax_tensor = sess.graph.get_tensor_by_name('import/colors_prob/colors_prob/Identity:0')
    x = np.random((1, 128, 128, 3))

    predictions = sess.run(softmax_tensor, {'import/conv2d_input:0': x})
    frozen_graph2 = freeze_session(sess, output_names=['import/colors_prob/colors_prob/Identity'])

    print("OK")