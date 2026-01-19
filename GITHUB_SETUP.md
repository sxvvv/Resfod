# GitHub 上传指南

## 已准备的文件

以下核心代码和文档已经准备好提交：

### 核心代码
- `models/` - 模型架构代码
- `utils/` - 工具函数库

### 文档
- `README.md` - 项目说明文档（专业版）
- `METHODOLOGY.md` - 技术方法文档
- `CONTRIBUTING.md` - 贡献指南
- `requirements.txt` - 依赖包列表

### 已排除（在.gitignore中）
- `results/` - 训练结果和checkpoint
- `train_IR.py` - 训练脚本
- `inference.py` - 推理脚本
- `train*.sh` - 训练脚本
- `run_inference.sh` - 推理脚本
- `convert_allweather_to_lmdb.py` - 数据转换脚本

## 上传到GitHub的步骤

### 1. 提交代码到本地仓库

```bash
cd /home/suxin/resfod

# 查看将要提交的文件
git status

# 提交代码（已准备好的文件）
git add .gitignore README.md requirements.txt models/ utils/ METHODOLOGY.md CONTRIBUTING.md

# 创建提交
git commit -m "Initial commit: Add core ResFoD model implementation

- Add TrajCFMNet model architecture
- Add DegradationParser module
- Add FoD core algorithms
- Add utility functions (metrics, dataset, factor utils)
- Add comprehensive documentation"
```

### 2. 在GitHub上创建仓库

1. 登录 GitHub
2. 点击右上角的 "+" -> "New repository"
3. 填写仓库信息：
   - **Repository name**: `resfod` (或你喜欢的名字)
   - **Description**: "Compositional Flow Matching for Image Restoration"
   - **Visibility**: Public (或 Private，根据需要)
   - **不要**初始化 README、.gitignore 或 license（我们已经有了）
4. 点击 "Create repository"

### 3. 连接本地仓库到GitHub

```bash
# 添加远程仓库（替换 YOUR_USERNAME 为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/resfod.git

# 或者使用SSH（如果你配置了SSH密钥）
# git remote add origin git@github.com:YOUR_USERNAME/resfod.git

# 验证远程仓库
git remote -v
```

### 4. 推送代码到GitHub

```bash
# 推送代码到GitHub（第一次）
git push -u origin master

# 或者如果你的默认分支是main
git branch -M main
git push -u origin main
```

### 5. 验证上传结果

访问你的GitHub仓库页面，应该看到：
- ✅ README.md 显示在仓库首页
- ✅ models/ 和 utils/ 目录存在
- ✅ METHODOLOGY.md 文档存在
- ✅ .gitignore 正确排除不需要的文件
- ✅ 没有 train_IR.py、inference.py 等训练/推理脚本
- ✅ 没有 results/ 目录

## 后续更新

如果需要更新代码，使用以下命令：

```bash
# 查看修改
git status

# 添加修改的文件
git add models/your_file.py utils/your_file.py

# 提交
git commit -m "Description of changes"

# 推送
git push
```

## 注意事项

1. **不要提交** 训练脚本、推理脚本、results目录
2. **不要提交** 个人数据或checkpoint文件
3. 确保 `.gitignore` 正确配置
4. 在推送前检查 `git status` 确保没有意外添加文件

## 如果遇到问题

### 问题：推送被拒绝
```bash
# 如果是首次推送且GitHub仓库有内容，需要先拉取
git pull origin master --allow-unrelated-histories
git push -u origin master
```

### 问题：想移除已提交的文件
```bash
# 从git中移除但保留本地文件
git rm --cached file_name
git commit -m "Remove file_name"
git push
```
