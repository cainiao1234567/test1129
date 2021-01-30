import numpy as np

np.random.randint(0,9,size=(4,4,))
np.random.randint创建一个N行N列的数组，其中的范围可以通过前面两个参数来决定

使用函数生成特殊的数组
import numpy as np
a1 = np.zeros((2,2)) #生成一个所有元素都是0的2行2列的数组
a2 = np.ones((3,2)) #生成一个所有元素都是1的3行3列的数组
a3 = np.full((2,4),9) #生成一个所有元素都是9的2行2列的数组
a4 = np.eye(3) #生成一个在斜方形上元素为1，其他元素都为0的3×3矩阵

数组中的数据类型都是一致的，不能不同类型
b = np.array([1,2,3,4,5],dtype=np.int8)
print(b)
print(b.dtype)


f.astype('U')  #修改f的数据类型
narray.ndim   #检测数组的维度
a1.ndim  a1的维度

narray.shape # 数组的维度的元组 
a1.shape   #注意是元组

a1.reshape((x,y)) #将a1更改为x行y列的数组
a1.flatten() #将a1扁平化,就是所谓的转化成一维数组

ndarray.size : 获取数组中总的元素的个数
ndarray.itemsize : 数组中每个元素占的大小,单位是字节

reshape flatten 不会改变原来的值,会生成一个新的数组

多维数组的索引
也是通过中括号来索引和切片,在括号中,
使用逗号进行分割,逗号前面的是行,逗号后面的是列
多维数组中只有一个值,代表行
a[1]
a[1:3]   # 获取a的前两行
a[[0,2,3]] #获取第0行第2行第3行
a[1,1] # 获取a中的1行1列
a[[1,2],[4,5]] 获取第2行4列和第3行5列
#[1,2]代表的是行 [4,5]代表的是列
逗号前面的是行,逗号后面的是列
a[1:3,4:6] 获取的数据是2行4和5列和3行的4和5列

获取某一列的值
a[:,2] 获取第3列的值

值的替换

where函数

数组的广播机制
广播原则
如果两个数组的后缘维度(从末尾开始算起的维度)的轴长度相符或其中一方的长度为1,
则认为他们是广播兼容。广播会在缺失和或长度为1的维度上进行

flatten和ravel方法：
flatten是将数组转化为一维数组后，然后将这个拷贝返回回去，
所以后续对这个返回值修改不会影响之前的数据

ravel是将数组转化为一维数组后，将这个视图(可以理解为引用)返回回去，所以后续对这个返回值进行修改会影响之前的数组

数组的叠加：
vstack : 将数组按照垂直方向进行叠加，数组的列数必须相同
vstack3 = np.vstack([vstack1,vstack2])#必须加上[]
vstack1在vstack2的上面

hstack : 将数组按照水平方向进行叠加，数组的行数必须相等
hstack3 = np.hstack([hstack1,hstack2])#必须加上[]

concatenate([a,b],axis) : 将；两个数组进行叠加，
但是具体按照水平方向还是垂直方向，
则要看axis的参数，如果axis=0,则代表垂直方向，
如果axis=1,则代表水平方向，如果axis=None,
那么会将两个数组组合成一个一维数组


数组的切割
按照水平方向进行分割np.hsplit()
np.hsplit(hs1,想要分割的数) 
这个想要分割的数必须能被hs1的列数整除，否则会报错
np.hsplit(h1,(1,2))

按照垂直方向进行分割
np.vsplit(hs1,n(此时的n必须得被行数整除))

split 或者 array_split ： 用于指定切割方式,在切割的时候需要指定是按照行还是按照列	axis=1代表列，axis=0代表行

数组的转置和轴对换
t1 = np.random.randint(0,10,size=(3,4))
t2 = t1.T
t2为t1的转置 就是将行变成列

t1.dot(t2)
矩阵相乘

转置：
t1.transpose()

数组的拷贝
a = np.arange(12)
c = a.view() #浅拷贝

深拷贝
a = np.arange(12)
c = a.copy() #深拷贝

浅拷贝a is c返回的是True
深拷贝a is c返回的是False

ravel是浅拷贝
flatten是深拷贝

文件操作：

文件保存
np.savetxt(frame,array,fmt='%.18e',
delimiter=None,
header="这个是列的名称取决于delimiter",
comments='')
frame : 文件
array : 存入文件的数组
fmt : 写入文件的格式
delimiter : 分割字符串，默认是任何空格

help(np.savetxt)

读取文件
np.loadtxt(frame,dtype=np.float,delimiter=None,unpack=False)

frame : 文件
dtype : 数据类型
delimiter : 分隔符
skiprows : 跳过前面x行
usecols ： 读取指定的列,用元组组合
unpack : 如果True,读取出来的数组是转置的

np独有的存储解决方案
存储： np.save(fname,array) 或者 np.savez(fname,array)
其中,前者函数的扩展名是.npy后者的扩展名是.npz,后者是经过压缩的

加载: np.load(fname)

np.savetxt() 和 np.loadtxt()不能存储三维以上的数组
可以设置header

csv文件处理
import csv

with open('xxx.csv','r') as fp:
	reader = csv.reader()
	#reader是一个迭代器


csv.DictReader() # 字典的形式读取csv文件
使用csv.DictReader()不会包含带有标题的第一行
是一个迭代器

写入到csv文件
with open('./classroom.csv','w',newline='',encoding='utf-8') as fp:
	writer = csv.writer(fp)


以字典的形式写入到csv文件中
记得写入表头
header = ['username','age','height']
with open('./classroom1.csv','w',encoding='utf-8',newline='') as fp:
	writer = csv.DictWriter(fp,header) #header值得是文件的列名
	writer.writerheader() #将表头信息写入文件中
	writer.writerow(不是元组形式了，字典形式)或者writer.writerows(不是元组形式了，字典形式)


NAN : not a number 属于浮点类型
INF : infinity 代表无穷大的意思 也属于浮点类型

np.NAN 或者np.nan
删除缺失值
data[~np.isnan(data)]

使用delete方法删除指定的行,axis=0表示删除行,lines表示删除行的号
data1 = np.delelte(data,lines,axis=0)

np.where(np.isnan(data)) #返回data中nan的位置,以元组形式


除了delete用axis=0表示行以外,其他大部分函数都是axis=1来表示行
data.sum(axis=1)

NAN和所有的值进行计算结果都是NAN
NAN != NAN

np.delete()比较特殊，通过axis=0来代表行，而其他大部分函数是通过
axis=1来代表行



np.random模块

np.random.seed
用于指定

np.random.rand() 生成一个值为[0,1]之间的数组,
形状由参数决定,如果没有参数,那么将返回一个随机值,


np.random.randn()
np.random.randn(2,3) 生成一个2行3列的数组,数组中的值都要满足标准正态分布


np.random.randint()


np.random.choice() 从列表或者数组中,随机采样。或者是从指定的区间中进行采样
采样的个数可以通过参数指定

np.random.shuffle() 将原来的数组打乱
a = np.arange(10)
np.random.shuffle(a)

axis理解
操作方式: 如果指定轴进行相关的操作,那么他会使用轴下的每个直接子元素的第0个
第1个，第2个……分别进行相关的操作

np的通用函数 一元函数


np.rint() 或者np.round() 四舍五入

np.modf() : 返回两部分 一部分为整数部分,另一部分为小数部分

np.isnan() np.isinf()

a = np.random.uniform(-10,10,size=(3,5))
a[(a>0) & (a<5)] # 条件必须加上括号


聚合函数

np.sort()   排序函数 指定轴进行排序,默认是使用数组的最后一个轴进行排序
np.sort() 不会改变数组的本身
a.sort() 则会更改数组,此时的a已经变成了排序后的数组

np.argsort() 返回排序后的下标值


np.apply_along_axis: 沿着某个轴执行执行的函数是

np.logical_and 相当于 &
np.logical_or 相当于 | 

np.linspace() 用来指定区间内的值平均分成多少份

np.unique() 返回数组中唯一的值
np.unique(d,return_counts=True)
结果为两个数组,其中一个数组为数组中出现的数,另一个为数组中数出现的个数

pandas库
import pandas as pd
s1 = pd.Series([1,2,3])

pandas有两个最主要的也是最重要的数据结构,Series和DataFrame
Series是一种一维标记的数组型对象
由数据和索引组成
索引在左  数据在右

Series的创建
1.通过列表来创建
2.数组创建
arr1 = np.arange(1,6)
s2 = pd.Series(arr1,index=['a','b','c','d','e'])
s2.values
s2.index 
#3.通过字典来创建
dict = {'name':'李宁','age':18,'class':'三班'}
s3 = pd.Series(dict)
可以指定顺序,因为字典无序
s3 = pd.Series(dict,index=['name','age','class'])
s3

s3.isnull() 
s3.notnull()

通过索引获取数据
s3.index
s3.values

标签名
s3['age']

s3[['name','age']]


#标签切片包括最后一个 'class' 
s3[['name':'class']]

#布尔索引

s2.name = 'temp' 对象名
s2.index.name = 'year' 对象名索引

s2.head()  #默认前五行

s2.tail() #显示最后几行



DataFrame是一个表格形数据结构，
它含有一组有序的列，每列可以是不同类型的值

DataFrame既可以行索引也可以列索引

数组列表或元组构成的字典构造dataframe

frame.columns

索引对象不可变,保证了数据安全

索引的基本操作 ： 重新索引 增 删 改 查
ps1.reindex() 重新索引

#行索引创建 ps1.reindex()
#列索引重建 ps1.reindex(columns=[]) 

s1 = pd.Series({'f':999})
ps3 = ps1.append(s1)
ps3


插入列
pd1.insert(0,'E',[9,99,999])
pd1

增加行
#标签索引loc
pd1.loc['d'] = [1,1,1,1,1]
pd1

或者
row = {'E':6,'A':6,'B':6,'C':6,4:6}
pd5 = pd1.append(row,ignore_index=True)
pd5

删
del ps1['d']

drop删除轴上数据

删除列时需要指定轴
pd1.drop('A',axis=1)

pd1.drop('A',axis='colums')

pd1.drop('c',inplace=True) #在原对象上删除,不会返回一个新对象

如果创建的DataFrame形式的,可以用pd1.A = 6 A这一列都为6

标签索引
pd1.loc['a'] 获取a这一行

pd1.loc['a','A'] = 1000  对a行A列的值修改为1000

标签索引或按照索引名都是包含终止索引的

不连续索引
ps1[['a','c','e']]

布尔索引

如果为DataFrame类型,pd1['A']['a']  A列a行的数据

DataFrame类型切片获取的是行

高级索引三部分
loc标签索引 iloc位置索引 ix标签与位置混合索引

ix标签已经弃用，了解即可

#第一个参数索引的是行,第二个参数索引的是列
pd1.loc['a':'b','A']



lianjia_df = pd.read_csv(r'lianjia.csv')

## pandas对齐运算

<img src="D:\data_learn\算术方法表.PNG" alt="算术方法表" style="zoom:80%;" />





s1.add(s2,fill_value=0)

#字母r开头会翻转参数
df1.rdiv(1)

<img src="D:\data_learn\sub用法示例.PNG" alt="sub用法示例" style="zoom:60%;" />



#通过apply将函数应用到列和行
f = lambda x:x.max()
df.apply(f)



#通过applymap将函数应用到行或列
f2 = lambda x:'%.2f'%x
df.applymap(f2)

s1.sort_index()   按照索引进行排序 ,默认为升序指的是索引的升序

s1.sort_index(ascending=False)   按照索引进行排序 ,默认为降序指的是索引的降序

如果为DataFrame类型

pd1.sort_index() #默认为行排序

pd1.sort_index(axis=1)  #列排序

s1.sort_values()#根据值的大小进行排序



pd2.sort_index(by=['a','c']) #指定多列进行排序



s2 = s1.unique()

s1.value_counts()  每个值出现的个数

s1.isin([8])  判断8是否存在

pds1.isnull() 是否存在缺失值

丢弃缺失数据 dropna()   pds.dropna() 默认丢弃行

df3.dropna(axis=1)

#填充缺失数据
df3.fillna(-1)

## 常见的Index种类

Index,索引

Int64Index层次索引

Multiindex层级索引

DatetimeIndex时间戳类型

#外层获取
s1['b']

交换  pd1.swaplevel()交换内层和外层的索引

s1.sum() #默认按照列求和

df.cumsum()  #累加和

df.describe()   #数据汇总

 























