import os
import uff
import subprocess
import tensorflow as tf


def load_pb_from_file(path_to_frozen_model):
    if path_to_frozen_model.startswith("~"):
        path_to_frozen_model = os.path.expanduser(path_to_frozen_model)
    with tf.gfile.GFile(path_to_frozen_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def change_file_extension(path_str: str, orig_ext: str = None, new_ext: str = None) -> str:
    """
    Make sure that a given path string would have the expected extension.
    Optionally remove a previous extension if requested.
    Example: change_file_extension('my_picuture.tiff', '.tiff', '.tif') -> 'my_picuture.tif'
    @param path_str: The file path to manipulate
    @param orig_ext: The file extension to remove (if specified) including the '.', i.e. '.tiff'
    @param new_ext: The new file extension to append (if specified) including the '.', i.e. '.tif'
    @return: The manipulated string as described above.
    """
    extention_to_trim = None if orig_ext == '' else orig_ext
    extention_to_add = '' if new_ext is None else new_ext
    trimmed_path = path_str.rsplit(extention_to_trim,1)[0]
    appended_ext = '{path}{ext}'.format(path=trimmed_path, ext=extention_to_add)
    return appended_ext


def pb_to_uff(pb_file='input.pb', uff_file=None):
    if pb_file.startswith("~"):
        pb_file = os.path.expanduser(pb_file)
    if not uff_file:
        uff_file = change_file_extension(pb_file, '.pb', '.uff')
    pb_graph_def = load_pb_from_file(pb_file)
    ret = uff.from_tensorflow(pb_graph_def, output_filename=uff_file, return_graph_info=True)
    #ret = uff.from_tensorflow(pb_graph_def, list_nodes=True)
    print("Model converted to file {}".format(uff_file))
    fsz=os.stat(uff_file).st_size
    bsz=len(ret[0])
    print("Model size validation ('File size' == 'Buffer size'): {}".format(fsz == bsz))
    print("File size: {file_sz}, Buffer size: {buff_sz}".format(file_sz=sizeof_fmt(fsz), buff_sz=sizeof_fmt(bsz)))
    return ret


def sizeof_fmt(num, suffix='B'):
    numf = float(num)
    units = ['', 'K', 'M', 'G']
    for ex, unit in enumerate(units):
        div = 2 ** (10*(ex))
        if abs(numf) < 1024.0 * div:
            return "%3.1f%s%s" % (numf/div, unit, suffix)
    return "%.1f%s%s" % (numf, 'T', suffix)


def run_trtexec(trtexecpath='/usr/src/tensorrt/bin/trtexec', uffFilePath='None', uffInputNodeDefs=[],
                uffOutputNodeDefs=[], saveEngineToFile=None):
    """
    Use trtexec to optimize your generated uff mode and save it to a TensorRT Engine file (also known as plan file)
    @param trtexecpath: The path of a binary executable of trtexec to use (default: /usr/src/tensorrt/bin/trtexec)
    @param uffFilePath: The path of a uff model file to optimize
    @param uffInputNodeDefs: A list of tf.NodeDef input nodes as returned from pb_to_uff()
    @param uffOutputNodeDefs: A list of tf.NodeDef output nodes as returned from pb_to_uff()
    @param saveEngineToFile: If specified, the generated engine will be saved to a plan file of the specified path.
    @return: None
    """
    # verify args:
    if not os.path.isfile(trtexecpath) or not os.access(trtexecpath, os.X_OK):
        raise Exception("trtexecpath={} must ba a path to an executable file".format(trtexecpath))
    if not os.path.isfile(uffFilePath):
        raise Exception("uffFilePath={} must ba a path to an existing file".format(uffFilePath))
    if not uffInputNodeDefs:
        raise Exception("The list of uffInputNodeDefs must contain at least one element")
    if not uffOutputNodeDefs:
        raise Exception("The list of uffOutputNodeDefs must contain at least one element")

    # build the command:
    progbin = [trtexecpath]
    uff_file_arg = ["--uff={}".format(uffFilePath)]
    graph_input_args = [as_uff_input_arg(n) for n in uffInputNodeDefs]
    graph_output_args = [as_uff_output_arg(n) for n in uffOutputNodeDefs]
    save_engine_arg = [] if saveEngineToFile is None else ['--saveEngine={}'.format(saveEngineToFile)]
    all_args = progbin + uff_file_arg + graph_input_args + graph_output_args + save_engine_arg
    subprocess.check_call(args=all_args)

    # """
    #='/usr/src/tensorrt/bin/trtexec'
    # #/usr/src/tensorrt/bin/trtexec --uff=/home/eyalper/projects/cppFlowATR/graphs/cm/output_graph_08_03_20.uff --uffInput=conv2d_input_3,3,128,128 --output=k2tfout_0 --saveEngine=cm.engeine
    # @return: exit code of trtexec
    # """


def as_uff_input_arg(node: tf.NodeDef) -> str:
    """
    Convert a given tf.NodeDef of a graph input to a string in order to provide to trtexec as a command line argument
    @param node: A tf.NodeDef object with the relevant data
    @return: A string with the form of '--uffInput=<tensor_name>,<C>,<H>,<W>'
    """
    name = node.name
    shape = node.attr['shape'].shape.dim
    N, H, W, C = [d.split('size:')[-1].strip() for d in str(shape).splitlines() if 'size:' in d]
    return '--uffInput={in_name},{c},{h},{w}'.format(in_name=name, c=C, h=H, w=W)


def as_uff_output_arg(node: tf.NodeDef) -> str:
    """
    Convert a given tf.NodeDef of a graph output to a string in order to provide to trtexec as a command line argument
    @param node:  A tf.NodeDef object with the relevant data
    @return:  A string with the form of '--output=<tensor_name>'
    """
    name = node.name
    return '--output={}'.format(name)


if __name__ == '__main__':
    # path_to_pb = '/home/eyalper/projects/cppFlowATR/graphs/frozen_inference_graph_humans.pb'
    # path_to_pb = '/home/eyalper/projects/cppFlowATR/graphs/cm/output_graph_08_03_20.pb'
    path_to_pb = 'outColorNetOutputs_08_03_20_full/k2tf/output_graph.pb'
    path_to_uff = change_file_extension(path_to_pb, '.pb', '.uff')
    print('*' * 20 + ' Converting to UFF ' + '*' * 20)
    ret, inputs, outputs = pb_to_uff(path_to_pb)
    print('*' * 20 + '**** Done ...  ****' + '*' * 20)
    print('input Node(s): {}'.format([node.name for node in inputs]))
    print('output Node(s): {}'.format([node.name for node in outputs]))
    print('*'*25 + ' Running trtexec though it is obviously going to fail ....' + '*'*25)
    run_trtexec(uffFilePath=path_to_uff, uffInputNodeDefs=inputs, uffOutputNodeDefs=outputs)