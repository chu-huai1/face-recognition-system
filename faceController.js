// 顶部统一引入模型
const Face = require('../models/FaceModel');
const faceService = require('../services/faceService');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// 配置文件上传
const uploadDir = './uploads/temp';
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const upload = multer({
  dest: uploadDir,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
  fileFilter: (req, file, cb) => {
    const filetypes = /jpeg|jpg|png/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = filetypes.test(file.mimetype);
    
    if (extname && mimetype) {
      cb(null, true);
    } else {
      cb(new Error('只允许上传 JPG、JPEG 或 PNG 格式的图片'));
    }
  }
});

// 识别人脸
const recognizeFace = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: '请上传人脸图片' });
    }

    const result = await faceService.recognizeFace(
      req.file.path,
      req.body.recognitionType
    );

    // 安全删除临时文件
    fs.unlink(req.file.path, (err) => {
      if (err) console.error('删除临时文件失败:', err);
    });

    // 加固：失败强制显示未识别
    if (!result.success) {
      if (result.data) {
        result.data.name = "未识别";
        result.data.similarity = "0%";
      }
    }

    return res.status(result.success ? 200 : 400).json(result);
  } catch (error) {
    console.error('识别错误:', error);
    return res.status(500).json({
      success: false,
      message: '服务器错误：识别失败',
      data: { name: "未识别", similarity: "0%" }
    });
  }
};

// 1:1 人脸验证
const verifyFaces = async (req, res) => {
  try {
    if (!req.files || req.files.length < 2) {
      return res.status(400).json({
        success: false,
        message: '请上传两张人脸图片'
      });
    }

    const threshold = req.body.threshold || 80;
    const result = await faceService.verifyFaces(
      req.files[0].path,
      req.files[1].path,
      threshold
    );

    // 清理临时文件
    req.files.forEach(file => {
      fs.unlink(file.path, err => err && console.error('删除临时文件失败:', err));
    });

    return res.status(result.success ? 200 : 400).json(result);
  } catch (error) {
    console.error('验证错误:', error);
    return res.status(500).json({ success: false, message: '验证失败' });
  }
};

// 添加人脸到库
const addFace = async (req, res) => {
  try {
    const { name, employeeId } = req.body;

    if (!name || !employeeId) {
      return res.status(400).json({
        success: false,
        message: '姓名和员工ID不能为空'
      });
    }

    if (!req.file) {
      return res.status(400).json({ success: false, message: '请上传人脸图片' });
    }

    const result = await faceService.addFaceToDatabase(
      name,
      employeeId,
      req.file.path
    );

    // 清理临时文件
    fs.unlink(req.file.path, err => err && console.error('删除临时文件失败:', err));

    return res.status(result.success ? 201 : 400).json(result);
  } catch (error) {
    console.error('添加人脸错误:', error);
    return res.status(500).json({ success: false, message: '注册失败' });
  }
};

// 获取所有人脸
const getAllFaces = async (req, res) => {
  try {
    const faces = await Face.find().select('name employeeId imageUrl gender age createdAt');
    return res.status(200).json({
      success: true,
      count: faces.length,
      data: faces
    });
  } catch (error) {
    console.error('获取列表错误:', error);
    return res.status(500).json({ success: false, message: '获取人脸列表失败' });
  }
};

// 删除人脸
const deleteFace = async (req, res) => {
  try {
    const face = await Face.findById(req.params.id);
    if (!face) {
      return res.status(404).json({ success: false, message: '未找到该人脸' });
    }

    await Face.findByIdAndDelete(req.params.id);
    return res.status(200).json({ success: true, message: '人脸删除成功' });
  } catch (error) {
    console.error('删除错误:', error);
    return res.status(500).json({ success: false, message: '删除失败' });
  }
};

module.exports = {
  upload,
  recognizeFace,
  verifyFaces,
  addFace,
  getAllFaces,
  deleteFace
};