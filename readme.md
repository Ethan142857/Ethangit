# 多模态融合模型实现
这是我的多模态融合模型的存储库

## 设置

该模型的实现基于Python3.运行代码前，你需要安装以下依赖：

- torch==1.13.1

- torchvision==0.14.1

-torchaudio==0.13.1

-pandas==1.4.2

-numpy==1.21.5

-scikit-learn==1.0.2

-transformers==4.18.0

-Pillow==9.0.1

您只需运行

```python
pip install requirements.txt
```

## 文件结构
```python
|-- Final_Lab
    |-- Fusion_model.py # 多模态融合模型的实现代码
    |-- text_only.py #消融实验结果中仅文本模型的实现代码
    |-- image_only.py #消融实验结果中仅图像模型的实现代码
    |-- train.txt #训练集数据
    |-- test_without_label.txt #测试集数据
    |-- test_with_label.txt #多模态融合模型的预测结果
    |-- test_with_label_text_only.txt #仅文本模型的预测结果
    |-- test_with_label_image_only.txt #仅图像模型的预测结果
```

## 运行代码
1.进入Lab目录，所使用的数据已经预先下载好
```python
cd Final_Lab
```

2.运行任何你想运行的代码，如：
```python
python Fusion_model.py
```