
## markdown 单元格还支持 LaTex 语法
获得下面的 LaTex 方程式：
$$\int_0^{+\infty} x^2 dx$$

## Magic functions



```python
%lsmagic
```




    Available line magics:
    %alias  %alias_magic  %autocall  %automagic  %autosave  %bookmark  %cd  %clear  %cls  %colors  %config  %connect_info  %copy  %ddir  %debug  %dhist  %dirs  %doctest_mode  %echo  %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %macro  %magic  %matplotlib  %mkdir  %more  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %popd  %pprint  %precision  %profile  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %ren  %rep  %rerun  %reset  %reset_selective  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%cmd  %%debug  %%file  %%html  %%javascript  %%js  %%latex  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.




```python
%pwd
```




    'C:\\Users\\Administrator\\Documents\\mnist'



## 使用Matplotlib绘图

在Jupyter Notebook中，如果使用Matplotlib绘图，有时是弹不出图像框的，此时，可以在开头加入


```python
%matplotlib inline
```

## 使用连接

[link](#link_name)

<a id ='link_name'></a>

## Introduction

content

## 超链接

欢迎来到[百度一下](http://blog.leanote.com/post/freewalk/Markdown-%E8%AF%AD%E6%B3%95%E6%89%8B%E5%86%8C)

## 无序列表

 - 列表1 
 - 列表2 
 
 + 列表3
 + 列表4
 
 * 列表5
 * 列表6

## 有序列表
1. 有序列表1
2. 有序列表2

## 列表缩进
* 轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。轻轻的我走了，正如我轻轻的来。

## 内部缩进
* 1 DeepLearning
> 监督学习
> 监督学习

## 插入图像
* 行内式 
> #### 百度logo
![百度logo](https://ss0.bdstatic.com/5aV1bjqh_Q23odCf/static/superman/img/logo/bd_logo1_31bdc765.png "Baidu logo")
* 参考式
> #### 百度logo
![百度logo][logo]
[logo]: https://ss0.bdstatic.com/5aV1bjqh_Q23odCf/static/superman/img/logo/bd_logo1_31bdc765.png

## LaTeX 公式
* 表示行内公式 $ E=mc^2 $
* 表示整行公式
 $$\sum_{i=1}^n a_i=0$$  
 
 $$f(x_1,x_x,\ldots,x_n) = x_1^2 + x_2^2 + \cdots + x_n^2 $$  
  
 $$\sum^{j-1}_{k=0}{\widehat{\gamma}_{kj} z_k}$$
 
 $$\int_0^{+\infty} x^2 dx$$  
 
 访问 [MathJax](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference) 参考更多使用方法

## 路程图

flow 

## 表格
序号 | 姓名|年龄
-|-|-
1|小明|12
2|小花|11

## 分割线
***
项目名称:
***
项目内容：

## 插入代码


## 内置 html代码

<table>
    <tr>
        <th rowspan="2">值班人员</th>
        <th>星期一</th>
        <th>星期二</th>
        <th>星期三</th>
    </tr>
    <tr>
        <td>李强</td>
        <td>张明</td>
        <td>王平</td>
    </tr>
</table>


```python

```
