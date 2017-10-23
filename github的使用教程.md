
# Git 和 github的使用教程

1 注册github账号  
2 下载 git客户端  
3 打开 Git Bash  
    3.1 git config --global user.name "Cncqleeds"  
    3.2 git config --global user.email "cncqleeds@yahoo.com"  
4 设置SSH  
    4.1  ssh-keygen -t rsa -C  cncqleeds@yahoo.com  
    4.2  copy "id_rsa.pub"文件到github账户下  
5 测试是否成功  
     ssh -T git@github.com   
6 在github上创建一个远程库 repository    
    6.1  创建 deeplearning_with_python    
7 创建本地库    
    7.1  在 git文件夹下，创建一个同名文件夹 deeplearning_with_python    
    7.2  右击文件夹，在弹出菜单中选择git bash ，弹出命令行窗口，输入：git init 创建本地库    
8 绑定本地库和远程库    
    git remote add origin git@github.com Cncqleeds/deeplearning_with_python.git   
9 管理本地文件    
    9.1 创建 test.txt 文件     
    9.2 查看 git的状态 git status  
    9.3 添加要管理的文件 git add test.txt  
    9.4 保存当前版本 git commit -m "新建test.txt"  
    9.5 查看log记录 git log  
    9.6 任意 编辑 test.txt  
    9.7 循环9.2 - 9.6  
    9.8 回退到以前的某个版本 git reset --hard  
10 暂存区  

    
    
## 借助github 托管项目代码

### 基本概念
1 仓库 Repository  
2 收藏 Star   
3 复制克隆项目 Fork   
4 发起请求 Pull Request  
5 关注 Watch  
6 事务卡片 Issue    
7 Github主页   
8 仓库主页    
9 个人主页  


### 基本操作
1 Create new file  
2 Upload files  
3 Find file  
4 Clone or download  
5 new issue and close issue

### git 命令
1 ls   
2 pwd  
3 mkdir src  
4 touch hello_world.py  
5 rm hello_world.py  
6 git rm hello.txt  删除暂存区的文件
7 git add hello_world.py  添加到暂存区
9 git commit -m "添加 hello_world.py 到仓库"  提交到仓库 
10 git status
11 vi hello_world.py 编辑文件  
12 esc 退出编辑
13 进入编辑  i or a or o三种方式
14 :wq 保存编辑
15 rm -rf hello_world.py 删除本地文件  
16 git config --list  显示配置信息  
17 clear  清空bash  


### 远程github  
1 设置SSH  
    1.1  ssh-keygen -t rsa -C  cncqleeds@yahoo.com  
    1.2  copy "id_rsa.pub"文件到github账户下 
2 git clone git@github.com:Cncqleeds/deeplearning_with_python.git 克隆远程仓库  
3 Enter passphrase for key '/c/Users/Administrator/.ssh/id_rsa': ****
4 git push 提交到远程仓库




# -------------------------------------------------------------------------#


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

### 提交到本地仓库
git commit -m "描述"

### 提交到远程仓库
git push







```python

```
