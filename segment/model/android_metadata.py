import os
from absl import app
from absl import flags
import tensorflow as tf
import flatbuffers
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string("model_file", None,
                        "Path and file name to the TFLite model file.")
    flags.DEFINE_string("label_file", None, "Path to the label file.")
    flags.DEFINE_string("export_directory", None,
                        "Path to save the TFLite model files with metadata.")
    flags.mark_flag_as_required("model_file")
    flags.mark_flag_as_required("label_file")
    flags.mark_flag_as_required("export_directory")


class ModelSpecificInfo(object):
    """Holds information that is specificly tied to an pose estimation."""

    def __init__(self, name, version, image_width, image_height, image_min,
                 image_max, num_classes, author):
        self.name = name
        self.version = version
        self.image_width = image_width
        self.image_height = image_height
        self.image_min = image_min
        self.image_max = image_max
        # self.mean = mean
        # self.std = std
        self.num_classes = num_classes
        self.author = author


_MODEL_INFO = {
    "model.tflite":
        ModelSpecificInfo(
            name="Product Mark Model",
            version="v1",
            image_width=320,
            image_height=320,
            image_min=0,
            image_max=255,
            num_classes=1,
            author="Foreo")
}


class MetadataPopulatorForPoseEstimation(object):
    """Populates the metadata for an pose estimation."""

    def __init__(self, model_file, model_info, label_file_path):
        self.model_file = model_file
        self.model_info = model_info
        self.label_file_path = label_file_path
        self.metadata_buf = None

    def populate(self):
        """Creates metadata and then populates it for an pose rstimation."""
        self._create_metadata()
        self._populate_metadata()

    def _create_metadata(self):
        """Creates the metadata for an pose estimation."""

        # Creates model info.
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.model_info.name
        model_meta.description = (
            "Identify the 17 bone joints of the characters in the picture and draw them,and estimate the action based on the joint point coordinates")
        model_meta.version = self.model_info.version
        model_meta.author = self.model_info.author
        model_meta.license = ("Apache License. Version 2.0 "
                              "http://www.apache.org/licenses/LICENSE-2.0.")

        # Creates input info.
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = (
            "Input image to be identify. The expected image is {0} x {1}, with "
            "three channels (red, blue, and green) per pixel. Each value in the "
            "tensor is a single byte between {2} and {3}.".format(
                self.model_info.image_width, self.model_info.image_height,
                self.model_info.image_min, self.model_info.image_max))
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = (
            _metadata_fb.ColorSpaceType.RGB)
        input_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.ImageProperties)
        input_stats = _metadata_fb.StatsT()
        input_stats.max = [self.model_info.image_max]
        input_stats.min = [self.model_info.image_min]
        input_meta.stats = input_stats

        # Creates output info.
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "probability"
        output_meta.description = "Probabilities of the %d labels respectively." % self.model_info.num_classes
        output_meta.content = _metadata_fb.ContentT()
        output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
        output_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties)
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(self.label_file_path)
        label_file.description = "Labels for actions that the model can detect."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        output_meta.associatedFiles = [label_file]

        # Creates subgraph info.
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(
            model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        self.metadata_buf = b.Output()

    def _populate_metadata(self):
        """Populates metadata and label file to the model file."""
        populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
        populator.load_metadata_buffer(self.metadata_buf)
        populator.load_associated_files([self.label_file_path])
        populator.populate()


def main(_):
    model_file = FLAGS.model_file
    model_basename = os.path.basename(model_file)
    if model_basename not in _MODEL_INFO:
        raise ValueError(
            "The model info for, {0}, is not defined yet.".format(model_basename))
    export_model_path = os.path.join(FLAGS.export_directory, model_basename)
    # Copies model_file to export_path.

    # 定义要读取的源文件路径
    output_path = "./model_with_metadata/best-fp16.tflite"

    # 定义目标文件路径
    input_path = "./best-fp16.tflite"

    # 使用tf.io.read_file()函数读取源文件内容
    with tf.io.gfile.GFile(input_path, 'rb') as input_file:
        # 读取文件内容
        content = input_file.read()

    # 创建输出文件并写入内容（使用Unicode编码）
    with tf.io.gfile.GFile(output_path, 'wb') as output_file:
        output_file.write(content)

    print("成功将文件从{}复制到{}".format(input_path, output_path))


    #tf.io.gfile.copy(model_file, export_model_path, overwrite=False)
    # Generate the metadata objects and put them in the model file
    export_model_path = "./model_with_metadata"
    populator = MetadataPopulatorForPoseEstimation(
        export_model_path, _MODEL_INFO.get(model_basename), FLAGS.label_file)
    populator.populate()

    # Validate the output model file by reading the metadata and produce
    # a json file with the metadata under the export path
    displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
    export_json_file = os.path.join(FLAGS.export_directory, os.path.splitext(model_basename)[0] + ".json")
    json_file = displayer.get_metadata_json()
    with open(export_json_file, "w") as f:
        f.write(json_file)

    print("Finished populating metadata and associated file to the model:")
    print(model_file)
    print("The metadata json file has been saved to:")
    print(export_json_file)
    print("The associated file that has been been packed to the model is:")
    print(displayer.get_packed_associated_file_list())


if __name__ == "__main__":
    define_flags()
    app.run(main)