
# Github 使用教程

### 在Github上创建同本地同名的一个项目仓库


### 从Github上下载该项目仓库
git clone git@github.com:Cncqleeds/deeplearning_with_python.git

### 在下载好的本地项目库下面创建文件
touch test.txt

### 编辑文件
vi test.txt

### 退出编辑
esc + :wq

### 添加到暂存区
git add test.txt
git add *

### 提交到本地仓库
git commit -m "描述"

### 提交到远程仓库
git push
git push -f

## 在实际开发中遇到以下两种情景

### 刚一提交 commit 到仓库，突然想起漏掉两个文件还没有添加 add
git reset  
git checkout -- filename 恢复删除的文件 
git checkout branchname 切换分支名
git branch newname 创建分支 


### 刚一提交commit 到仓库，突然想起版本描述写的不全面，无法彰显本次修改的重大意义...
git commit --amend   
git commit --amend -m"new desc"   
git rm --cached filename 只删除缓存区的文件  
git mv oldfilename newfilename 修改文件名
