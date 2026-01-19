# 🔐 解决GitHub认证问题

## 问题
```
fatal: Authentication failed for 'https://github.com/sxvvv/Resfod.git/'
remote: No anonymous write access.
```

## 解决方案

### 方法1：使用Personal Access Token（推荐）

GitHub不再支持使用密码推送，需要使用Personal Access Token。

#### 步骤1：创建Personal Access Token

1. **访问**：https://github.com/settings/tokens
2. **点击** "Generate new token" → "Generate new token (classic)"
3. **填写信息**：
   - **Note**: `resfod-push`（或任何描述）
   - **Expiration**: 根据需要选择（推荐90天或No expiration）
   - **权限**：勾选 `repo` 权限（这会自动选择所有子权限）
4. **点击** "Generate token"
5. **复制Token**：显示一串类似 `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` 的字符串
   - ⚠️ **重要**：这个Token只会显示一次，请立即复制保存

#### 步骤2：使用Token推送

```bash
cd /home/suxin/resfod

# 推送时，用户名输入你的GitHub用户名
# 密码输入刚才复制的Personal Access Token
git push -u origin master
```

**提示**：
- Username: 你的GitHub用户名（如 `sxvvv`）
- Password: 粘贴你的Personal Access Token（`ghp_xxxxx...`）

#### 步骤3：保存凭证（可选）

避免每次推送都输入Token：

```bash
# 使用Git Credential Manager（推荐）
git config --global credential.helper store

# 或者使用缓存（15分钟内有效）
git config --global credential.helper 'cache --timeout=900'
```

### 方法2：使用SSH密钥（推荐长期使用）

#### 步骤1：检查是否已有SSH密钥

```bash
ls -la ~/.ssh
```

如果有 `id_rsa.pub` 或 `id_ed25519.pub`，跳到步骤3。

#### 步骤2：生成SSH密钥

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

按回车使用默认路径，可以设置密码或直接回车。

#### 步骤3：复制公钥

```bash
cat ~/.ssh/id_ed25519.pub
# 或者
cat ~/.ssh/id_rsa.pub
```

复制整个输出内容。

#### 步骤4：添加到GitHub

1. 访问：https://github.com/settings/keys
2. 点击 "New SSH key"
3. **Title**: `My Computer`（或任何描述）
4. **Key**: 粘贴刚才复制的公钥
5. 点击 "Add SSH key"

#### 步骤5：更改远程仓库地址为SSH

```bash
cd /home/suxin/resfod

# 删除HTTPS方式的远程仓库
git remote remove origin

# 添加SSH方式的远程仓库（替换YOUR_USERNAME）
git remote add origin git@github.com:sxvvv/Resfod.git

# 验证
git remote -v

# 推送（不需要输入密码）
git push -u origin master
```

### 方法3：使用GitHub CLI（最简单）

```bash
# 安装GitHub CLI（如果还没安装）
# Ubuntu/Debian:
# sudo apt install gh

# 或者：
# curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
# sudo apt update
# sudo apt install gh

# 登录GitHub
gh auth login

# 选择：
# - GitHub.com
# - HTTPS
# - 授权GitHub CLI

# 然后推送
git push -u origin master
```

## 验证远程仓库地址

检查远程仓库地址是否正确：

```bash
git remote -v
```

应该看到：
```
origin  https://github.com/sxvvv/Resfod.git (fetch)
origin  https://github.com/sxvvv/Resfod.git (push)
```

如果地址不对，可以修改：

```bash
# 修改远程仓库地址
git remote set-url origin https://github.com/sxvvv/Resfod.git

# 或者使用SSH
git remote set-url origin git@github.com:sxvvv/Resfod.git
```

## 快速解决方案

**最简单的方法**：使用Personal Access Token

1. 创建Token：https://github.com/settings/tokens（选择`repo`权限）
2. 推送时使用Token代替密码：
   ```bash
   git push -u origin master
   # Username: sxvvv
   # Password: <粘贴你的Token>
   ```

## 测试连接

```bash
# 测试HTTPS连接
git ls-remote https://github.com/sxvvv/Resfod.git

# 测试SSH连接（如果使用SSH）
git ls-remote git@github.com:sxvvv/Resfod.git
```
