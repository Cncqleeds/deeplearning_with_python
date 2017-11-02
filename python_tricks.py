
# coding: utf-8

# ## python 编程中一些常用的小技巧
# 

# In[1]:

# 1. 原地交换两个数
x,y = 3, 5
print(x,y)
x,y = y,x
print(x,y)


# In[2]:

# 2. 序列数据的倒置
test_list = [1,2,3]
print(test_list)
test_list.reverse()
print(test_list)
test2_list = test_list[::-1]
print(test2_list)
test_str = "Hello world"
test2_str = test_str[::-1]
print(test_str,test2_str)


# In[3]:

# 3. 利用列表中的所有元素，创建一个字符串 "".join
str_list = ["利用","列表","中","的","all","elements"]
test3_str = " ".join(str_list)
print(test3_str)

test_set = {"Hello","world","what","are","you","doing"}
print(test_set)

print(" ".join(test_set)) # 注意 set是 无序的


# In[4]:

# 4. lambda 表达式 
test_dict = {
    "add": lambda x,y:x+y,
    "sub": lambda x,y:x-y
}
print(test_dict["add"](3,5))


# In[5]:

# 5. * 展开
def f(a,b,c):
    
    print(a+b+c)

test_list = ['hello',"world","!"]
print(test_list)
# f(test_list) # 报错
f(*test_list)

test_dict = {"x":1,"y":2,"z":3}
print(test_dict)
f(*test_dict)
# f(**test_dict) # 报错


# In[6]:

# 6. python 中的几种类型
test_range = range(5)
print(type(test_range))
test_list = list(test_range)
print(test_list)


# In[7]:

# 7. python 中的一些常用内置函数
# dir(__builtin__)


# In[8]:

# 8. functools 工具包
import functools as fts
# fts.reduce()


# In[9]:

# 9. 链状比较操作符
n = 3
test_bool = 1 <= n <= 10
print(test_bool)


# In[10]:

# 10. 使用三元操作符来进行条件赋值
y = 19
x = 10 if y is 9 else 20
print(x)

class classA:
    def __init__(self,name = "",age = 0):
        self.name = name
        self.age  = age
    def show(self):
        print("name:%s, age:%d."%(self.name,self.age))
        
class classB:
    def __init__(self,name = "",sex = ""):
        self.name = name
        self.sex  = sex
    def show(self):
        print("name:%s, sex:%s."%(self.name,self.sex))
        

pram1 = None
pram2 = None
    
obj = (classA if y is 19 else classB)(pram1,pram2)


# In[11]:

# 11. 计算三个数中的最小值
def small(a,b,c):
    return a if a<b and a<c else(b if b<a and b<c else c)
print(small(1,2,3))
# print(small(1,1,0))
# print(small(1,2,3))
# print(small(1,1,1))
# print(small(4,2,3))
# print(small(1,4,1))
# print(small(1,5,3))
# print(small(1,6,0))
# print(small(1,6,1))
# print(small(1,0,1))


# In[12]:

# 12. 列表推导

test_list =[m**2 if m>10 else m**4 for m in range(20)]
print(test_list)


# In[13]:

# 13. 多行字符串
mult_str =("Hello world"
          "Python 编程"
          "2017-11-02")
print(mult_str)


# In[14]:

# 14. 存储列表元素到变量中
test_list =[1,2,3,4,5]
a,b,c,d,e = test_list
print(a,b,c,d,e)


# In[15]:

# 15.  _ 操作符
# _1+1

# print(_)


# In[16]:

# 16. 查询导入模块的路径
import functools

print(functools)


# In[17]:

# 17. 字典/集合推导
test_dict = {i:i*i for i in range(10)}
print(test_dict)
test_set = {i*2 for i in range(10)}
print(test_set)


# In[18]:

# 18 调试脚本
# import pdb
x = 1
y = 2
# pdb.set_trace()
z = x + y
print(z)


# In[19]:

# 19. in 操作符 (关系操作符 in  is  not < <= !=)
test_set ={"Apple","Google","Baidu","Facebook"}
test_str ="Baidu"
print("yes") if test_str in test_set else print("No")

print(0 is not 1)
    


# In[20]:

# 20. 序列切片
test_list = list(range(10))
print(test_list)
print(test_list[8:1:-2])


# In[21]:

# 21. 序列翻转
test_dict = {i:i*2 for i in range(10)}
test_list = [1,2,3,4]
# r_test_dict = dict(reversed(test_dict))
r_test_list = list(reversed(test_list))
print(r_test_list)
print(test_dict)
# print(r_test_dict)


# In[22]:

# 22. 玩转枚举
test_list = [10,20,30]
print(test_list)
test_dict = dict(enumerate(test_list))
print(test_dict)

a,b,c = enumerate(test_list)
print(a,b,c)
print(type(a))

w,x,y,z = range(4)
print(x,y,z)
print(type(x))


# In[23]:

# 23. 使用一行代码计算任何数的阶乘
# reduce(func,list,num) ,func必须是二元操作函数,list 是一个列表,num是其实操作数

import functools as fts
result = (lambda k:fts.reduce(int.__mul__,range(1,k+1),1))(10)
print(result)


# In[24]:

# 24. 找出列表中出现最频繁的数
# count 计数
# max(iteration,key = func)  最大值
test_list = [1,2,2,3,4,1,0,1,1,2,4,4,2,3,1,0,4,2,0,3,1,2,0,4,2]
# test_set = set(test_list)
print(max(set(test_list),key = test_list.count))
print(test_list.count(5))


# In[25]:

# 25. 重置递归限制
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(999)
print(sys.getrecursionlimit())
sys.setrecursionlimit(1000)
print(sys.getrecursionlimit())


# In[26]:

# 26. 检查对象的内存使用情况
a = [1,2,3,4,8,1]
import sys
print(sys.getsizeof(a))


# In[27]:

# 27. 使用lambda 表达式模仿输出
# map(func,iters)
import sys
lprint = lambda *args: sys.stdout.write(" -> ".join(map(str,args))+"\n")
lprint(1,"hello world",'1',1.0)
print(1,"hello",'1',1.0)
print([1,"hello",'1',1.0])


# In[28]:

# 28. 从两个相关的序列构建一个字典
test_tuple1 = (1,2,3)
test_tuple2 = ('a','b','z')
# zip 方法
test_dict = dict(zip(test_tuple1,test_tuple2))
print(test_dict)


# In[29]:

# 29. 一行代码实现搜索字符串的多个前后缀
print("http://www.baidu.com".startswith(("http://","https://")))
print("http://www.baidu.com".endswith((".com",".cn")))


# In[30]:

# 30. 不适用循环构造一个列表
import itertools

test_list = [[1,2],[3,4],[0,1]]
new_list = itertools.chain.from_iterable(test_list)
print(list(new_list))


# In[31]:

# 31. 在python中实现一个真正的switch-case 语句
# switch(n)
# {
#     case n1:
#         code
#         break
#     case n2:
#         code
#         break
#     ...
#     default:
#         code  
# }

