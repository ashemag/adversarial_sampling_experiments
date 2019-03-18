from collections import OrderedDict
from collections import defaultdict

d = OrderedDict(defaultdict(lambda : None))

d['item1'] = 0
d['item2'] = 3

print(d)