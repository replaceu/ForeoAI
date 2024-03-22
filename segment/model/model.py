import json
import tflite_model_maker
from absl import logging
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import tensorflow as tf
assert  tf.__version__.startswith('2')
tf.get_logger().setLevel("ERROR")
logging.set_verbosity(logging.ERROR)

#指定模型
#spec = model_spec.get('efficientdet_lite0')
print('--------------------------------------------------------------')
spec = tflite_model_maker.object_detector.EfficientDetLite0Spec(
uri = './pretraining/efficientdet_lite0_feature-vector_1'
)
print('数据集划分需要读取多幅图像，可能花费几分钟时间，请耐心等待...')
train_data,validation_data,test_data = object_detector.DataLoader.from_csv('./dataset/datasets.csv')

print(train_data)
print('开始模型训练...')

#训练模型，指定模型参数
model = object_detector.create(train_data,
                               model_spec =spec,
                               validation_data =validation_data,
                               epochs=30,
                               batch_size=16,
                               train_whole_model = True)
#将训练好的模型导出为TFLite model，并保存到当前工作目录下
print('训练结束！导出模型....')
model.export(export_dir='.')
model.summary()

# #保存与模型输出一致的标签列表
# classes = ['???']*model.model_spec.config.num_classes
# label_map = model.model_spec.config.label_map
#
# for label_id,label_name in label_map.as_dict().items():
#     classes[label_id-1] = label_name
# print(classes)
#
# #模型标签保存到文件
# with open('labels.txt',w) as f:
#     for i in range(len(classes)):
#         for label in classes:
#             f.write(label+"\r")
#
# #在测试集上评测训练好的模型
# #dict_new  = {}
# print('开始在测试集上对计算机版模型评估...')
# dict_local = model.evaluate(test_data,batch_size = 16)
# print(f'计算机版本模型在测试集上评估结果：\n {dict_local}')
# #加载TFLite格式的模型，在测试集上评估
# print('开始在测试集上对优化后的TFLite模型评估...')
# dict_tfilte = model.evaluate_tflite('model.tflite',test_data)
# print(f'优化后的TFLite模型在测试集上评估结果：\n {dict_tfilte}')
#
# #保存模型的评估结果
# for key in dict_local:
#     dict_local[key] = str(dict_local[key])
#     print(f'{key}:{dict_local[key]}')
#
# with open('dict_local.txt','w') as f:
#     f.write(json.dumps(dict_local))
#
# for key in dict_tfilte:
#     dict_tfilte[key] = str(dict_tfilte[key])
#     print(f'{key}:{dict_tfilte[key]}')
#
# with open('dict_tfilte.txt','w') as f:
#     f.write(json.dumps(dict_tfilte))


