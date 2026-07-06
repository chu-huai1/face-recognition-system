const mongoose = require('mongoose');

const faceSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  employeeId: {
    type: String,
    required: true,
    unique: true  // 唯一索引，防止重复注册
  },
  featureVector: {
    type: [Number],  // 存储人脸特征数组（128/512维）
    required: true
  },
  imageUrl: {
    type: String,
    required: true
  },
  gender: {
    type: String,
    enum: ['男', '女', 'unknown'],  // 与后端统一
    default: 'unknown'
  },
  age: {
    type: Number,
    min: 0,
    max: 120
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// 每次保存自动更新时间
faceSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// 索引：加速姓名/工号查询
faceSchema.index({ name: 1 });

module.exports = mongoose.model('Face', faceSchema);