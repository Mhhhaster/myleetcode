## argparse



## args和kwargs

```python
def test_kwargs(first, *args, **kwargs): #args是元组tuple，用*解包；kwargs是字典，用**解包
   print('Required argument: ', first)
   print(type(kwargs))
   for v in args:
      print ('Optional argument (args): ', v)
   for k, v in kwargs.items():
      print ('Optional argument %s (kwargs): %s' % (k, v))

test_kwargs(1, 2, 3, 4, k1=5, k2=6)
```



```text
Required argument:  1
<class 'dict'>
Optional argument (args):  2
Optional argument (args):  3
Optional argument (args):  4
Optional argument k1 (kwargs): 5
Optional argument k2 (kwargs): 6
```