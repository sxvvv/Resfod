# 📤 上传到GitHub的完整步骤

## ✅ 第一步：代码已提交到本地仓库

代码已经成功提交到本地Git仓库。现在需要上传到GitHub。

---

## 🌐 第二步：在GitHub创建新仓库

### 2.1 登录GitHub并创建仓库

1. **打开浏览器**，访问 https://github.com
2. **登录**你的GitHub账号
3. **点击右上角的 "+" 按钮**，选择 **"New repository"**

### 2.2 填写仓库信息

**Repository name**: `resfod` （或你喜欢的名字，如 `compositional-flow-matching`）

**Description**: 
```
Compositional Flow Matching for Image Restoration - Decomposes complex combined degradations into atomic factors
```

**Visibility**:
- ✅ **Public** - 公开（推荐，更多人可以看到和引用）
- ⭕ **Private** - 私有（如果需要保密）

**重要：不要勾选以下选项**：
- ❌ Add a README file（我们已经有了）
- ❌ Add .gitignore（我们已经有了）
- ❌ Choose a license（可以后续添加）

### 2.3 创建仓库

点击绿色的 **"Create repository"** 按钮

---

## 🔗 第三步：连接本地仓库到GitHub

### 3.1 添加远程仓库

在终端执行以下命令（**替换 `YOUR_USERNAME` 为你的GitHub用户名**）：

```bash
cd /home/suxin/resfod

# 使用HTTPS方式（推荐，简单）
git remote add origin https://github.com/YOUR_USERNAME/resfod.git

# 或者使用SSH方式（如果你配置了SSH密钥）
# git remote add origin git@github.com:YOUR_USERNAME/resfod.git
```

**如何找到你的GitHub用户名？**
- 在GitHub右上角头像旁边就是你的用户名
- 或者访问 https://github.com/settings/profile 查看

### 3.2 验证远程仓库

```bash
git remote -v
```

应该看到：
```
origin  https://github.com/YOUR_USERNAME/resfod.git (fetch)
origin  https://github.com/YOUR_USERNAME/resfod.git (push)
```

---

## 🚀 第四步：推送代码到GitHub

### 4.1 检查分支名称

```bash
git branch
```

如果显示 `* master`，使用：
```bash
git push -u origin master
```

如果显示 `* main`，使用：
```bash
git push -u origin main
```

### 4.2 推送代码

**如果是 master 分支：**
```bash
git push -u origin master
```

**如果是 main 分支：**
```bash
# 可能需要先重命名分支
git branch -M main
git push -u origin main
```

### 4.3 输入GitHub凭证

如果是第一次推送，GitHub会要求验证身份：

**选项1：使用Personal Access Token（推荐）**
- 如果提示输入密码，使用你的 **Personal Access Token**（不是GitHub密码）
- 如何创建：https://github.com/settings/tokens
- 权限选择：`repo` 权限

**选项2：使用GitHub CLI**
```bash
gh auth login
git push -u origin master
```

**选项3：配置SSH密钥（推荐用于长期使用）**
- 参考：https://docs.github.com/en/authentication/connecting-to-github-with-ssh

---

## ✅ 第五步：验证上传结果

### 5.1 访问GitHub仓库

在浏览器打开：`https://github.com/YOUR_USERNAME/resfod`

### 5.2 检查上传的文件

应该看到：
- ✅ **README.md** - 显示在仓库首页
- ✅ **models/** 目录 - 包含模型架构代码
- ✅ **utils/** 目录 - 包含工具函数
- ✅ **METHODOLOGY.md** - 技术文档
- ✅ **requirements.txt** - 依赖包列表
- ✅ **.gitignore** - Git忽略规则

### 5.3 确认排除的文件

**不应该看到**：
- ❌ `train_IR.py`
- ❌ `inference.py`
- ❌ `results/` 目录
- ❌ `train*.sh` 脚本
- ❌ `run_inference.sh`

---

## 🔄 后续更新代码

如果需要更新代码，使用以下命令：

```bash
# 1. 查看修改
git status

# 2. 添加修改的文件
git add models/your_file.py utils/your_file.py README.md

# 3. 提交
git commit -m "Update: description of changes"

# 4. 推送
git push
```

---

## ❓ 常见问题

### Q1: 推送时提示 "fatal: remote origin already exists"

**解决**：
```bash
# 删除旧的远程仓库
git remote remove origin

# 重新添加
git remote add origin https://github.com/YOUR_USERNAME/resfod.git
```

### Q2: 推送时提示认证失败

**解决**：
- 使用Personal Access Token代替密码
- 或者配置SSH密钥

### Q3: 推送时提示 "Updates were rejected"

**解决**：
```bash
# 如果GitHub仓库有内容（比如自动生成的README），先拉取
git pull origin master --allow-unrelated-histories

# 解决可能的冲突后，再推送
git push -u origin master
```

### Q4: 忘记添加某个文件到.gitignore

**解决**：
```bash
# 从git中移除但保留本地文件
git rm --cached file_name

# 提交
git commit -m "Remove file_name from git"

# 推送
git push
```

---

## 📝 快速命令总结

```bash
# 1. 进入项目目录
cd /home/suxin/resfod

# 2. 添加远程仓库（替换YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/resfod.git

# 3. 推送代码
git push -u origin master
```

**完成！** 🎉
