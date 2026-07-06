const mongoose = require('mongoose');

const recordSchema = new mongoose.Schema({
  // 关联人脸表（必须）
  faceId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Face',
    required: true
  },
  // 姓名
  name: {
    type: String,
    required: true
  },
  // 工号
  employeeId: {
    type: String,
    required: true
  },
  // 人脸识别相似度
  similarity: {
    type: Number,
    required: true
  },
  // 抓拍图片地址
  imageUrl: {
    type: String,
    required: true
  },
  // 可选属性
  gender: String,
  age: Number,
  glasses: Boolean,
  emotion: String,

  // 识别耗时（毫秒/秒）
  recognizeCost: Number,

  // 识别来源：上传 / 摄像头
  recognitionType: {
    type: String,
    enum: ['upload', 'camera'],
    default: 'upload'
  },
  
  // 识别时间（标准时间格式）
  recognitionTime: {
    type: Date,
    default: Date.now
  },
  
  // 创建时间
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// 索引：加速查询（非常重要）
recordSchema.index({ employeeId: 1 });
recordSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Record', recordSchema);