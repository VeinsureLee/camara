---
title: 远程视频监控系统PC端数据处理
author: Veinsure Lee
date: 2025-04-20
---

<!-- 居中标题（HTML 实现） -->
<div style="text-align: center;">

# 远程视频监控系统PC端数据处理

</div>

<!-- 作者和日期（可选居中） -->
<div style="text-align: center;">

*作者：Veinsure Lee*  
*日期：2025-04-20*

</div>

---

## 一、引言


- **总项目要求**：
  1. 实验实现的功能：实时远程摄像监控，24 小时记录摄像机范围内的各种事件。
  2. 建议设备：嵌入式物联网应用层网关及智能执行机构系统 
  3. 功能模块划分
     - 摄像采集模块
     - 云台控制模块
     - 数据网络互传模块
     - 数据存储模块
     - 图像显示模块
     - 其他
  4. 完成指标要求
     - 间隔 10 分钟调整一次云台取一次数据保存
     - 当发生异常时，系统提出报警

<p style="text-indent: 2em;">
基于总项目要求，本项目解决接收esp32端发送的数据，并接入模型，
根据场景的不同进行<strong>人物识别</strong>或<strong>车牌识别</strong>。

</p>

<p style="text-indent: 2em;">
起步采用esp32cam进行数据的发送（UDP->TCP），后续框架将改为树莓派摄像头。
框架图如下：
</p>

<div align="center">
  <img src="demo/pic/construct/esp32视频监控.png" alt="演示" width="400">
</div>

<div align="center">
  <img src="demo/pic/construct/raspberrypi.png" alt="演示" width="400">
</div>

---

## 二、框架设计

### 1. 场景区分

- **家居场景（人物识别）**：
  1. 基于<strong>yolov8n</strong>训练的人物位置识别模型 
  2. 基于face recognize设计的人脸识别
  3. 基于mediapipe获取骨架数据训练模型实现动作识别
     - 爬虫爬取数据（简单分类）
     - mediapipe处理图片获取骨架数据
     - 训练神经网络完成分类人物

- **公共场景（车牌识别）**：
  - 色块提取初步识别车牌位置
  - 训练模型，对初步识别的车牌进行判断
  - 对确定的车牌位置进行二值化
  - 字符分割（计算列统计特性，均值卷积平滑，峰谷切割）

### 2. 项目进展
#### 1、视频接收
<div align="center">
  <img src="demo/gif/demo_pyside_show.gif" alt="演示" width="400">
</div>

#### 2、人类位置识别打开
<div align="center">
  <img src="demo/gif/demo_human_detect.gif" alt="演示" width="400">
</div>

#### 3、人脸识别打开

<div align="center">
  <img src="demo/gif/demo_face_detect.gif" alt="演示" width="400">
</div>

#### 4、动作识别打开

<div align="center">
  <img src="demo/gif/demo_action_detect.gif" alt="演示" width="400">
</div>

#### 5、车牌识别

<div align="center">
  <img src="demo/gif/demo_car.gif" alt="演示" width="400">
</div>

<div align="center">
  <img src="demo/gif/demo_car_plate_detect.gif" alt="演示" width="400">
</div>


### 3. 代码与引用
```python
# 代码块示例
def hello():
    print("Hello, Markdown!")

<div style="text-align: center;">

# 远程视频监控系统

</div>









