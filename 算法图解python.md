##第一章 二分查找  
二分查找的输入必须是有序链表。使用二分查找时，最多需要检查logn个元素，因为2的n次方为元素个数。  
可将元素存储在一系列相邻的桶中。  
```python
def binary_search(list, item):
  low = 0
  high = len(list)-1
  
  while low<=high:
  mid = (low+high)/2
  guess = list[mid]
  if guess == item:
    return mid
  if guess>item:
    high = mid-1
  else:
    low = mid+1
  return None

```
##第二章 数组和链表  
##第三章 递归  
##第四章 问题解决技巧：分而治之  
##第九章 问题解决技巧：动态规划  
如果没有高效的解决方案，使用：  
##第八章 贪婪算法  

当需要解决问题时，首先想到是否可以用散列表或用图来建立模型。  
##第五章 散列表  
##第六章 图算法：广度优先搜索  
##第七章 图算法：迪克斯特拉算法  

简单的机器学习算法：  
##第十章 KNN算法  
可用于创建推荐系统、OCR引擎、预测股价或其他值，或对物件进行分类。  
